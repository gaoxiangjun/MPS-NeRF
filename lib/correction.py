from torch.nn.modules.module import T
from lib.run_nerf_helpers import PositionalEncoding
import torch.nn as nn
import spconv
import torch.nn.functional as F
import torch
import cv2
# from psbody.mesh import Mesh
from lib.encoder import *
# from h36m_blend_weights import *
from lib.base_utils import read_pickle
from pytorch3d.ops.knn import knn_points
from memory_profiler import profile
from lib.transformer import Transformer
# nerf_input_ch = 71 # 39 x + 32 3d + 128 2d
# nerf_input_ch_2 = 384 # 128 2d + 256 alpha feature + 27 view + 32 3d


class DeformField(nn.Module):

    def __init__(self, D=8, W=256, input_ch=3, output_ch=3, skips=[4], deform_type="weights"):
        super(DeformField, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        
        self.pts_time_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        self.output_linear = nn.Linear(W, output_ch)
        if deform_type=='correction':
            torch.nn.init.constant_(self.output_linear.weight, 0)
            torch.nn.init.constant_(self.output_linear.bias, 0)
        self.deform_type = deform_type
    
    def forward(self, x):
        input_point_time = x
        h = input_point_time
        for i, l in enumerate(self.pts_time_linears):
            h = self.pts_time_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_point_time, h], -1)

        outputs = self.output_linear(h)
        if self.deform_type=='weights':
            # outputs = F.softmax(outputs, dim=0)
            outputs = F.softmax(outputs, dim=1) # right but not work
            # outputs = F.relu(outputs)

        return outputs


class CorrectionByf3d(nn.Module):
    def __init__(self, use_agg=False, human_sample=False, density_loss=False, with_viewdirs=False, use_f2d=True, smooth_loss=False, use_trans=False):
        super(CorrectionByf3d, self).__init__()
        self.encoder_3d = SparseConvNet(num_layers=2)
        self.encoder_2d = SpatialEncoder(num_layers=2)
        self.use_agg = use_agg
        self.with_viewdirs = with_viewdirs
        self.forward_deform = DeformField(D=3, input_ch=(143 + 128 if use_agg else 143), output_ch=3, deform_type='correction')
        self.backward_deform = DeformField(D=3, input_ch=(111 + 128 if use_agg else 111), output_ch=3, deform_type='correction')
        self.pos_enc = PositionalEncoding(num_freqs=6)
        self.view_enc = PositionalEncoding(num_freqs=4)
        self.SMPL_NEUTRAL = read_pickle(os.path.join('msra_h36m', 'smplx/smpl/SMPL_NEUTRAL.pkl'))
        W = 256
        self.image_shape = torch.zeros((2,))
        # self.register_buffer("image_shape", torch.empty(2), persistent=False)

        # xc + f3d + f2d
        self.actvn = nn.ReLU()
        self.skips = [4]
        nerf_input_ch = 71 if (not use_f2d) else (71+128)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(nerf_input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + nerf_input_ch, W) for i in range(7)])
        self.alpha_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)
        nerf_input_ch_2 = 384
        self.views_linear = nn.Linear((nerf_input_ch_2+27) if self.with_viewdirs else nerf_input_ch_2, W//2)
        self.rgb_linear = nn.Linear(W//2, 3)
        self.human_sample = human_sample
        self.density_loss = density_loss
        self.use_f2d = use_f2d
        self.smooth_loss = smooth_loss
        self.use_trans = use_trans
        self.transformer = None if not use_trans else Transformer(128)

    def projection(self, query_pts, R, T, K):
        # https://github.com/pengsida/if-nerf/blob/492e765423fcbfb9aeca345286a49074cc4e5a90/lib/utils/render_utils.py#L45
        r_t = np.concatenate([R, T], 2)
        # lower_row = np.array([[0., 0., 0., 1.]])
        # lower_row = np.expand_dims(lower_row, 0).repeat(4,axis=0)        
        # r_t = np.concatenate([r_t, lower_row], 1)
        RT = torch.from_numpy(r_t).cuda().float()

        # https://github.com/pengsida/if-nerf/blob/8b98eadd3aef6af23088ce3a9ee0504410fbf9e6/lib/utils/base_utils.py#L17
        xyz = torch.repeat_interleave(query_pts.unsqueeze(dim=0), repeats=RT.shape[0], dim=0)
        xyz = torch.bmm(xyz.float(), RT[:, :, :3].transpose(1, 2).float()) + RT[:, :, 3:].transpose(1, 2).float()
        xyz = torch.bmm(xyz, torch.from_numpy(K).cuda().float().transpose(1, 2).float())
        xy = xyz[:, :, :2] / (xyz[:, :, 2:] + 1e-5)

        return xy
    
    def encode_images(self, images):
        self.images = images
        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        self.encoder_2d(images)
    
    def window_feature(self, uv, window_width=5):
        # uv.shape torch.Size([4, 16384, 2])
        # shift_x = [ i - (window_width-1) / 2 for i in list(range(window_width))]
        # shift_y = [ i - (window_width-1) / 2 for i in list(range(window_width))]
        shift_x = [-2,-1,0,1,2]
        shift_y = [-2,-1,0,1,2]
        features = []
        
        for dx in shift_x:
            for dy in shift_y:
                new_uv = torch.stack([uv[:,:,1] + torch.tensor(dx).cuda().float(), uv[:,:,0] + torch.tensor(dy).cuda().float()],dim=-1)
                # [4, 512, 16384] (B, C, N)
                point_2d_feature = self.encoder_2d.index(new_uv, self.image_shape)
                features.append(point_2d_feature)

        # [4, 512, 16384] (B, C, N)
        features = torch.stack(features, dim=0)  # [25, 4, 512, 16384] (25, B, C, N)
        mean_feature_per_view = torch.mean(features, dim=0) # [4, 512, 16384]
        mean_feature = torch.mean(mean_feature_per_view, dim=0) # [512, 16384]
        mean_feature = mean_feature.transpose(0, 1)# [16384, 512]
        
        return mean_feature

    def coarse_deform_target2c(self, params, vertices,  query_pts):

        # joints transformation
        A, R, Th, joints = get_transform_params(self.SMPL_NEUTRAL, params)        
        J = np.hstack((joints, np.ones([joints.shape[0], 1])))
        self.c_joints = joints # smpl space
        self.t_A = A # target to canonical space transformation
        self.t_joints = np.matmul(A, J.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]

        # transform smpl vertices from world space to smpl space
        smpl_pts = torch.mm(torch.from_numpy((vertices - Th)).cuda().float(), torch.from_numpy(R).cuda().float())
        _, vert_ids, _ = knn_points(query_pts.unsqueeze(0).float(), smpl_pts.unsqueeze(0).float(), K=1)
        # bweights = self.SMPL_NEUTRAL['weights'][vert_ids.squeeze(0).cpu().numpy()]
        # bweights = torch.from_numpy(bweights).cuda().float().view(-1,24)
        vert_ids = vert_ids.detach()
        bweights = torch.from_numpy(self.SMPL_NEUTRAL['weights'][vert_ids.squeeze(0).cpu().numpy()]).cuda().float().view(-1,24)

        # From smpl space target pose to smpl space canonical pose
        A = torch.mm(bweights, torch.from_numpy(self.t_A.reshape(24, -1)).cuda().float())
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = query_pts - A[:, :3, 3]
        R_inv = torch.inverse(A[:, :3, :3].float())
        can_pts = torch.sum(R_inv * can_pts[:, None], dim=2)
        
        # From T To Big Pose        
        import copy
        big_pose_params = copy.deepcopy(params)
        big_pose_params['poses'] = np.zeros((1,72))
        big_pose_params['poses'][0, 5] = 45/180*np.pi
        big_pose_params['poses'][0, 8] = -45/180*np.pi
        big_pose_params['poses'][0, 23] = -30/180*np.pi
        big_pose_params['poses'][0, 26] = 30/180*np.pi
        A, R, Th, joints = get_transform_params(self.SMPL_NEUTRAL, big_pose_params)
        J = np.hstack((self.c_joints, np.ones([self.c_joints.shape[0], 1])))
        self.c_joints = np.matmul(A, J.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        A = torch.mm(bweights, torch.from_numpy(A.reshape(24, -1)).cuda().float())
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = torch.sum(A[:, :3, :3] * can_pts[:, None], dim=2)
        can_pts = can_pts + A[:, :3, 3]

        return can_pts

    def coarse_deform_c2source(self, params, t_vertices, query_pts):
        # Find nearest smpl vertex
        smpl_pts = torch.from_numpy(t_vertices).cuda().float()
        _, vert_ids, _ = knn_points(query_pts.unsqueeze(0).float(), smpl_pts.unsqueeze(0).float(), K=1)
        # bweights = self.SMPL_NEUTRAL['weights'][vert_ids.squeeze(0).cpu().numpy()]
        # bweights = torch.from_numpy(bweights).cuda().float().view(-1,24)
        vert_ids = vert_ids.detach()
        bweights = torch.from_numpy(self.SMPL_NEUTRAL['weights'][vert_ids.squeeze(0).cpu().numpy()]).cuda().float().view(-1,24)

        ### To Big Pose
        import copy
        big_pose_params = copy.deepcopy(params)
        big_pose_params['poses'] = np.zeros((1,72))
        big_pose_params['poses'][0, 5] = 45/180*np.pi
        big_pose_params['poses'][0, 8] = -45/180*np.pi
        big_pose_params['poses'][0, 23] = -30/180*np.pi
        big_pose_params['poses'][0, 26] = 30/180*np.pi
        A, R, Th, joints = get_transform_params(self.SMPL_NEUTRAL, big_pose_params)
        A = torch.mm(bweights, torch.from_numpy(A.reshape(24, -1)).cuda().float())
        A = torch.reshape(A, (-1, 4, 4))
        query_pts = query_pts - A[:, :3, 3]
        R_inv = torch.inverse(A[:, :3, :3].float())
        query_pts = torch.sum(R_inv * query_pts[:, None], dim=2)

        # get tar_pts, smpl space source pose
        A, R, Th, joints = get_transform_params(self.SMPL_NEUTRAL, params)
        self.s_A = A
        A = torch.mm(bweights, torch.from_numpy(self.s_A.reshape(24, -1)).cuda().float())
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = torch.sum(A[:, :3, :3] * query_pts[:, None], dim=2)
        smpl_src_pts = can_pts + A[:, :3, 3]
        
        # transform points from the smpl space to the world space
        R_inv = torch.inverse(torch.from_numpy(R).cuda().float())
        world_src_pts = torch.mm(smpl_src_pts, R_inv) + torch.from_numpy(Th).cuda().float()
        
        return smpl_src_pts, world_src_pts

    def forward_fusion(self, query_pts, params, point_3d_feature, agg_feature=None):
        """(x, x-joint_t, target_f_3d) → deformation Net 1 → X_c"""
        # if self.smooth_loss:
        #     query_pts.requires_grad_()
        query_pts_code = self.pos_enc(query_pts)
        pts = torch.repeat_interleave(query_pts.unsqueeze(1), 24, dim=1) # torch.Size([16384, 24, 3])
        t_joints = torch.from_numpy(self.t_joints).cuda().float() #  torch.Size([24, 3]
        diff = torch.reshape(pts - t_joints, [pts.shape[0], 72] ) # N * 72
        diff = diff#.detach()
        if agg_feature != None:
            f = torch.cat([query_pts_code, diff, point_3d_feature, agg_feature], dim=-1) # torch.Size([16384, 669])
        else:
            f = torch.cat([query_pts_code, diff, point_3d_feature], dim=-1) # torch.Size([16384, 669])
        return f
    
    def backward_fusion(self, canonical_pts, params, agg_feature=None):
        """(X_c, Pose_s, shape, x-joint_c, f_agg) → deformation Net 2 → X_s"""
        # canonical_pts.requires_grad = True
        canonical_pts_code = self.pos_enc(canonical_pts)
        pts = torch.repeat_interleave(canonical_pts.unsqueeze(1), 24, dim=1) # torch.Size([16384, 24, 3])
        c_joints = torch.from_numpy(self.c_joints).cuda().float() #  torch.Size([24, 3]
        diff = torch.reshape(pts - c_joints, [pts.shape[0], 72] ) # N * 72
        diff = diff#.detach()

        if agg_feature != None:
            f = torch.cat([canonical_pts_code, diff, agg_feature], dim=-1) # torch.Size([16384, 669])
        else:
            f = torch.cat([canonical_pts_code, diff], dim=-1) # torch.Size([16384, 669])

        return f

    def normalize_pts(self, source_pts, bounds):
        # bounds 2 * 3
        # source_pts N * 3
        mean_plus = 0.5 * (bounds[0] + bounds[1])
        mean_plus = torch.from_numpy(mean_plus).cuda().float()
        mean_sub = 0.5 * (bounds[1] - bounds[0])
        mean_sub = torch.from_numpy(mean_sub).cuda().float()

        normalized_pts = ((source_pts - mean_plus) / mean_sub)
        return normalized_pts

    def forward(self, sp_input, tp_input, query_pts, viewdir):
        # translate query points from world space target pose to smpl space target pose
        # Rh = tp_input['params']['Rh']
        # R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        R = torch.from_numpy( tp_input['params']['R']).cuda().float()
        Th = tp_input['params']['Th'].astype(np.float32)
        Th = torch.from_numpy(Th).cuda().float()
        query_pts = torch.mm(query_pts - Th, R)

        img_all = sp_input['img_all']
        img_all = torch.from_numpy(img_all).cuda().float()
        self.encode_images(img_all)
        
        if self.human_sample:
            tar_smpl_pts = torch.from_numpy(tp_input['vertices']).cuda().float()
            tar_smpl_pts = torch.mm(tar_smpl_pts - Th, R)
            distance, _, _ = knn_points(query_pts.unsqueeze(0).float(), tar_smpl_pts.unsqueeze(0).float(), K=1)
            distance = distance.squeeze(0).view(-1)
            pts_mask = torch.zeros_like(query_pts[:,0]).cuda().int()
            threshold = 0.05 ** 2
            pts_mask[distance < threshold] = 1
            
            zero_pts = torch.zeros(query_pts.shape).cuda().float()
            query_pts = query_pts[pts_mask==1]
            viewdir = viewdir[pts_mask==1]
            
        # get geometry aligned feature
        xyz_input_feature = torch.from_numpy(tp_input['feature']).cuda().float()
        voxel_integer_coord = torch.from_numpy(tp_input['coord']).cuda().float().int()
        coord = torch.zeros_like(voxel_integer_coord[:,0:1])
        voxel_integer_coord = torch.cat((coord, voxel_integer_coord), dim=1)
        out_sh = tp_input['out_sh']
        batch_size = 1
        sparse_smpl_vertices = spconv.SparseConvTensor(xyz_input_feature, voxel_integer_coord, out_sh, batch_size)
        normalized_source_pts = self.normalize_pts(query_pts, tp_input['bounds'])
        point_3d_feature_0 = self.encoder_3d(sparse_smpl_vertices, normalized_source_pts)

        # query points in smpl space source pose
        coarse_canonical_pts = self.coarse_deform_target2c(tp_input['params'], tp_input['vertices'], query_pts)        
        
        if self.use_agg:
            coarse_smpl_src_pts, coarse_world_src_pts = self.coarse_deform_c2source(sp_input['params'], sp_input['t_vertices'], query_pts=coarse_canonical_pts)#.detach()
            uv = self.projection(coarse_world_src_pts, sp_input['R_all'], sp_input['T_all'], sp_input['K_all'])
            agg_feature = self.window_feature(uv) # 512 * N
        else:
            agg_feature = None
        
        # (X_t, pose_t, shape, x-joint_t, f_agg) → deformation Net 1 → X_c
        fused_feature = self.forward_fusion(query_pts, tp_input['params'], point_3d_feature_0, agg_feature)
        correction = self.forward_deform(fused_feature.float())
        # coarse_canonical_pts = query_pts # for X posed
        # correction = torch.zeros_like(coarse_canonical_pts)
        canonical_pts = coarse_canonical_pts + correction

        if self.density_loss:
            can_smpl_pts = torch.from_numpy(tp_input['t_vertices']).cuda().float()
            distance, _, _ = knn_points(coarse_canonical_pts.unsqueeze(0).float(), can_smpl_pts.unsqueeze(0).float(), K=1)
            distance = distance.squeeze(0).view(-1)
            pts_mask = torch.zeros_like(query_pts[:,0]).cuda().int()
            threshold = 0.05 ** 2
            pts_mask[distance < threshold] = 1
        
        # (X_c, Pose_s, shape, x-joint_c, f_agg) → deformation Net 2 → X_s
        coarse_smpl_src_pts, _ = self.coarse_deform_c2source(sp_input['params'], sp_input['t_vertices'], query_pts=canonical_pts)
        fused_feature = self.backward_fusion(canonical_pts, sp_input['params'], agg_feature)
        correction_ = self.backward_deform(fused_feature.float())
        # correction_ = torch.zeros_like(coarse_smpl_src_pts)
        smpl_src_pts = coarse_smpl_src_pts + correction_

        # translate source points from smpl space source pose to world space source pose
        # Rh = sp_input['params']['Rh']
        # R = torch.tensor(cv2.Rodrigues(Rh)[0].astype(np.float32)).cuda().float()
        R = torch.tensor(sp_input['params']['R']).cuda().float()
        R_inv = torch.inverse(R)
        Th = torch.tensor(sp_input['params']['Th'].astype(np.float32)).cuda().float()
        world_src_pts = torch.mm(smpl_src_pts, R_inv) + Th
        
        # get geometry aligned feature
        xyz_input_feature = torch.from_numpy(sp_input['t_feature']).cuda().float()
        voxel_integer_coord = torch.from_numpy(sp_input['t_coord']).cuda().float().int()
        coord = torch.zeros_like(voxel_integer_coord[:,0:1])
        voxel_integer_coord = torch.cat((coord, voxel_integer_coord), dim=1)
        out_sh = sp_input['t_out_sh']
        batch_size = 1
        sparse_smpl_vertices = spconv.SparseConvTensor(xyz_input_feature, voxel_integer_coord, out_sh, batch_size)
        normalized_source_pts = self.normalize_pts(canonical_pts, sp_input['t_bounds'])
        point_3d_feature = self.encoder_3d(sparse_smpl_vertices, normalized_source_pts)
        
        # get mean pixel-aligned feature four view
        uv = self.projection(world_src_pts, sp_input['R_all'], sp_input['T_all'], sp_input['K_all'])
        point_2d_feature_0 = self.encoder_2d.index(uv, self.image_shape)
        point_2d_feature_0 = point_2d_feature_0.transpose(1,2)
        if self.use_trans and point_2d_feature_0.shape[1]!=0:
            point_2d_feature = point_2d_feature_0.transpose(0,1)
            point_2d_feature = self.transformer(point_2d_feature)
            point_2d_feature_1 = point_2d_feature[:,0,:]
            point_2d_feature_2 = point_2d_feature[:,1,:]
        else:
            point_2d_feature = torch.mean(point_2d_feature_0, dim=0)
            point_2d_feature_1 = point_2d_feature
            point_2d_feature_2 = point_2d_feature
        # point_2d_feature = torch.sum(point_2d_feature, dim=0)

        # Run in Nerf
        x = self.pos_enc(canonical_pts)
        if self.use_f2d:
            x = torch.cat((x, point_3d_feature, point_2d_feature_1), dim=1)
        else:
            x = torch.cat((x, point_3d_feature), dim=1)
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        
        if self.with_viewdirs:
            viewdir = self.view_enc(viewdir)
            h = torch.cat([feature, viewdir, point_2d_feature_1], -1)
        else:
            h = torch.cat([feature, point_2d_feature_2], -1)

        h = self.views_linear(h)
        h = F.relu(h)
        rgb = self.rgb_linear(h)
        
        if self.density_loss:
            pts_mask = torch.unsqueeze(pts_mask, -1)
            raw = torch.cat([rgb, alpha, pts_mask], -1)
            others = torch.cat([query_pts, smpl_src_pts, correction, correction_], -1)
        elif self.human_sample:
            # ret_query_pts, ret_smpl_src_pts = torch.zeros(zero_pts.shape).cuda().float(), torch.zeros(zero_pts.shape).cuda().float()
            # ret_correction, ret_correction_ = torch.zeros(zero_pts.shape).cuda().float(), torch.zeros(zero_pts.shape).cuda().float()
            # ret_query_pts[pts_mask==1], ret_smpl_src_pts[pts_mask==1],  ret_correction[pts_mask==1], ret_correction_[pts_mask==1] = query_pts.detach(), smpl_src_pts.detach(), correction, correction_
            # others = torch.cat([ret_query_pts, ret_smpl_src_pts, ret_correction, ret_correction_], -1)
            raw = torch.zeros(zero_pts.shape[0],4).cuda().float()
            raw[pts_mask==1] = torch.cat([rgb, alpha], -1)
            raw[pts_mask==0] = -80 # softplus and sigmoid is outside
            raw = torch.cat([raw, torch.unsqueeze(pts_mask, -1)], -1)
            
            # if correction.shape[0] != 0 and correction.requires_grad and correction_.requires_grad and query_pts.requires_grad:
            if self.smooth_loss and correction.shape[0] != 0 :# and query_pts_.requires_grad:
                query_pts_ = query_pts.detach()
                query_pts_.requires_grad_()
                back_fused_feature = self.forward_fusion(query_pts_, tp_input['params'], point_3d_feature_0.detach(), agg_feature.detach())
                back_correction = self.forward_deform(back_fused_feature)
                
                grad_u = torch.autograd.grad(back_correction[:,0], [query_pts_], grad_outputs=torch.ones_like(back_correction[:,0]), create_graph=True)[0]
                grad_v = torch.autograd.grad(back_correction[:,1], [query_pts_], grad_outputs=torch.ones_like(back_correction[:,1]), create_graph=True)[0]
                grad_w = torch.autograd.grad(back_correction[:,2], [query_pts_], grad_outputs=torch.ones_like(back_correction[:,2]), create_graph=True)[0]
                grad_deform = torch.stack([grad_u,grad_v,grad_w],dim=2)
                grad_deform = grad_deform.norm(dim=-1)

                canonical_pts_ = canonical_pts.detach()
                canonical_pts_.requires_grad_()
                fused_feature_ = self.backward_fusion(canonical_pts_, sp_input['params'], agg_feature.detach())
                back_correction_ = self.backward_deform(fused_feature_)
                
                grad_x = torch.autograd.grad(back_correction_[:,0], [canonical_pts_], grad_outputs=torch.ones_like(back_correction_[:,0]), create_graph=True)[0]
                grad_y = torch.autograd.grad(back_correction_[:,1], [canonical_pts_], grad_outputs=torch.ones_like(back_correction_[:,1]), create_graph=True)[0]
                grad_z = torch.autograd.grad(back_correction_[:,2], [canonical_pts_], grad_outputs=torch.ones_like(back_correction_[:,2]), create_graph=True)[0]
                grad_deform_ = torch.stack([grad_x,grad_y,grad_z],dim=2)
                grad_deform_ = grad_deform_.norm(dim=-1)
                # gc.collect()
            else:
                grad_deform = torch.zeros_like(correction)
                grad_deform_ = torch.zeros_like(correction_)
            # others = torch.cat([query_pts.detach(), smpl_src_pts.detach(), correction, correction_, grad_deform, grad_deform_], -1)
            others = torch.cat([query_pts, smpl_src_pts, correction, correction_, grad_deform, grad_deform_], -1)

        else:
            pts_mask = torch.ones_like(query_pts[:,0]).cuda().int()
            pts_mask = torch.unsqueeze(pts_mask, -1)
            raw = torch.cat([rgb, alpha, pts_mask], -1)
            # raw = torch.cat([rgb, alpha], -1)
            others = torch.cat([query_pts, smpl_src_pts, correction, correction_], -1)
        
        return raw, others


class CorrectionByUvhAgg(nn.Module):
    def __init__(self, use_agg=False):
        super(CorrectionByUvhAgg, self).__init__()
        self.encoder_3d = SparseConvNet(num_layers=2)
        self.encoder_2d = SpatialEncoder(num_layers=2)
        self.use_agg = use_agg
        self.forward_deform = DeformField(input_ch=(4 + 63 + 128 if use_agg else 4 + 63), output_ch=3, deform_type='correction')
        self.backward_deform = DeformField(input_ch=(1 + 63 + 128 if use_agg else 1 + 63), output_ch=3, deform_type='correction')
        self.pos_enc = PositionalEncoding(num_freqs=10)
        self.view_enc = PositionalEncoding(num_freqs=4)
        self.SMPL_NEUTRAL = read_pickle(os.path.join('msra_h36m', 'smplx/smpl/SMPL_NEUTRAL.pkl'))
        W = 256
        # self.register_buffer("image_shape", torch.empty(2), persistent=False)
        self.image_shape = torch.zeros((2,))

        # xc + f3d + f2d
        self.actvn = nn.ReLU()
        self.skips = [4]
        self.pts_linears = nn.ModuleList(
            [nn.Linear(nerf_input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + nerf_input_ch, W) for i in range(7)])
        self.alpha_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)
        self.views_linear = nn.Linear(nerf_input_ch_2, W//2)
        self.rgb_linear = nn.Linear(W//2, 3)
        self.pdist2 = nn.PairwiseDistance(p=2, keepdim=True)

    def projection(self, query_pts, R, T, K):
        r_t = np.concatenate([R, T], 2)
        RT = torch.from_numpy(r_t).cuda().float()

        xyz = torch.repeat_interleave(query_pts.unsqueeze(dim=0), repeats=RT.shape[0], dim=0)
        xyz = torch.bmm(xyz.float(), RT[:, :, :3].transpose(1, 2).float()) + RT[:, :, 3:].transpose(1, 2).float()
        xyz = torch.bmm(xyz, torch.from_numpy(K).cuda().float().transpose(1, 2).float())
        xy = xyz[:, :, :2] / (xyz[:, :, 2:] + 1e-10)

        return xy
    
    def encode_images(self, images):
        self.images = images
        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        self.encoder_2d(images)
    
    def window_feature(self, uv, window_width=5):
        # uv.shape torch.Size([4, 16384, 2])
        shift_x = [-2,-1,0,1,2]
        shift_y = [-2,-1,0,1,2]
        features = []
        
        # width*width   *   torch.Size([4, 16384, 2])
        for dx in shift_x:
            for dy in shift_y:
                new_uv = torch.stack([uv[:,:,1] + torch.tensor(dx).cuda().float(), uv[:,:,0] + torch.tensor(dy).cuda().float()],dim=-1)
                # [4, 512, 16384] (B, C, N)
                point_2d_feature = self.encoder_2d.index(new_uv, self.image_shape)
                features.append(point_2d_feature)

        # [4, 512, 16384] (B, C, N)
        features = torch.stack(features, dim=0)  # [25, 4, 512, 16384] (25, B, C, N)
        mean_feature_per_view = torch.mean(features, dim=0) # [4, 512, 16384]
        mean_feature = torch.mean(mean_feature_per_view, dim=0) # [512, 16384]
        mean_feature = mean_feature.transpose(0, 1)# [16384, 512]
        
        return mean_feature
    
    def coarse_deform_target2c(self, params, vertices,  query_pts):
        # joints transformation
        A, R, Th, joints = get_transform_params(self.SMPL_NEUTRAL, params)
        J = np.hstack((joints, np.ones([joints.shape[0], 1])))
        self.c_joints = joints # smpl space
        self.t_A = A # target to canonical space transformation
        self.t_joints = np.matmul(A, J.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]

        """
        faces = self.SMPL_NEUTRAL['f']
        # transform smpl vertices from the world space to the smpl space
        pxyz = np.dot(vertices - Th, R)
        # pxyz = vertices

        smpl_mesh = Mesh(pxyz, faces)
        # obtain the blending weights for query_pts in smpl space target pose
        closest_face, closest_points = smpl_mesh.closest_faces_and_points(pts)
        vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(
                                        closest_points, closest_face.astype('int32'))
        bweights = barycentric_interpolation(self.SMPL_NEUTRAL['weights'][vert_ids], bary_coords)
        bweights = torch.from_numpy(bweights).cuda().float().view(-1,24)
        """
        # transform smpl vertices from world space to smpl space
        smpl_pts = torch.mm(torch.from_numpy((vertices - Th)).cuda().float(), torch.from_numpy(R).cuda().float())
        _, vert_ids, _ = knn_points(query_pts.unsqueeze(0).float(), smpl_pts.unsqueeze(0).float(), K=1)
        bweights = self.SMPL_NEUTRAL['weights'][vert_ids.squeeze(0).cpu().numpy()]
        bweights = torch.from_numpy(bweights).cuda().float().view(-1,24)

        # translate query points from smpl space target pose to smpl space canonical pose
        A = torch.mm(bweights, torch.from_numpy(self.t_A.reshape(24, -1)).cuda().float())
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = query_pts - A[:, :3, 3]
        R_inv = torch.inverse(A[:, :3, :3].float())
        can_pts = torch.sum(R_inv * can_pts[:, None], dim=2)

        ### To Big Pose
        import copy
        big_pose_params = copy.deepcopy(params)
        big_pose_params['poses'] = np.zeros((1,72))
        big_pose_params['poses'][0, 5] = 45/180*np.pi
        big_pose_params['poses'][0, 8] = -45/180*np.pi
        big_pose_params['poses'][0, 23] = -30/180*np.pi
        big_pose_params['poses'][0, 26] = 30/180*np.pi
        A, R, Th, joints = get_transform_params(self.SMPL_NEUTRAL, big_pose_params)
        A = torch.mm(bweights, torch.from_numpy(A.reshape(24, -1)).cuda().float())
        A = torch.reshape(A, (-1, 4, 4))
        J = np.hstack((self.c_joints, np.ones([self.c_joints.shape[0], 1])))
        self.c_joints = np.matmul(A, J.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        can_pts = torch.sum(A[:, :3, :3] * can_pts[:, None], dim=2)
        can_pts = can_pts + A[:, :3, 3]

        return can_pts 

    def coarse_deform_c2source(self, params, t_vertices, query_pts):
        """
        faces = self.SMPL_NEUTRAL['f']
        # t_pose, smpl space
        pxyz = t_vertices
        smpl_mesh = Mesh(pxyz, faces)
        # obtain the blending weights for query_pts
        closest_face, closest_points = smpl_mesh.closest_faces_and_points(pts)
        vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(
                                        closest_points, closest_face.astype('int32'))
        bweights = barycentric_interpolation(self.SMPL_NEUTRAL['weights'][vert_ids], bary_coords)
        """

        smpl_pts = torch.from_numpy(t_vertices).cuda().float()
        _, vert_ids, _ = knn_points(query_pts.unsqueeze(0).float(), smpl_pts.unsqueeze(0).float(), K=1)
        bweights = self.SMPL_NEUTRAL['weights'][vert_ids.squeeze(0).cpu().numpy()]
        bweights = torch.from_numpy(bweights).cuda().float().view(-1,24)
        
        ### To Big Pose
        import copy
        big_pose_params = copy.deepcopy(params)
        big_pose_params['poses'] = np.zeros((1,72))
        big_pose_params['poses'][0, 5] = 45/180*np.pi
        big_pose_params['poses'][0, 8] = -45/180*np.pi
        big_pose_params['poses'][0, 23] = -30/180*np.pi
        big_pose_params['poses'][0, 26] = 30/180*np.pi
        A, R, Th, joints = get_transform_params(self.SMPL_NEUTRAL, big_pose_params)
        A = torch.mm(bweights, torch.from_numpy(A.reshape(24, -1)).cuda().float())
        A = torch.reshape(A, (-1, 4, 4))
        query_pts = query_pts - A[:, :3, 3]
        R_inv = torch.inverse(A[:, :3, :3].float())
        query_pts = torch.sum(R_inv * query_pts[:, None], dim=2)

        # get tar_pts, smpl space source pose
        A, R, Th, joints = get_transform_params(self.SMPL_NEUTRAL, params)
        self.s_A = A
        A = torch.mm(bweights, torch.from_numpy(self.s_A.reshape(24, -1)).cuda().float())
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = torch.sum(A[:, :3, :3] * query_pts[:, None], dim=2)
        smpl_src_pts = can_pts + A[:, :3, 3]

        # transform points from the smpl space to the world space
        R_inv = torch.inverse(torch.from_numpy(R).cuda().float())
        world_src_pts = torch.mm(smpl_src_pts, R_inv) + torch.from_numpy(Th).cuda().float()
        
        return smpl_src_pts, world_src_pts

    def forward_fusion(self, query_pts, vertices, vert_ids, t_vertices, agg_feature=None):
        # (X_t, pose_t, shape, x-joint_t, f_agg) → deformation Net 1 → X_c
        query_pts_code = self.pos_enc(query_pts)
        vert_ids = vert_ids.squeeze(0)
        # vert_ids_code = self.pos_enc(vert_ids.float())
        dist = self.pdist2(query_pts, torch.from_numpy(vertices).cuda().float()[vert_ids].squeeze(1))
        t_vertex = torch.from_numpy(t_vertices).cuda().float()[vert_ids].squeeze(1)
        # pts = torch.repeat_interleave(query_pts.unsqueeze(1), 24, dim=1) # torch.Size([16384, 24, 3])
        # t_joints = torch.from_numpy(self.t_joints).cuda().float() #  torch.Size([24, 3]
        # diff = torch.reshape(pts - t_joints, [pts.shape[0], -1] ) # N * 72
        # poses = torch.repeat_interleave(torch.from_numpy(params['poses']).cuda().float(), pts.shape[0], 0)
        # shapes = torch.repeat_interleave(torch.from_numpy(params['shapes']).cuda().float(), pts.shape[0], 0)

        if agg_feature != None:
            # f = torch.cat([query_pts_code, diff, poses, shapes, agg_feature], dim=-1) # torch.Size([16384, 669])
            f = torch.cat([query_pts_code, t_vertex, dist, agg_feature], dim=-1) # torch.Size([16384, 669])
            # f = torch.cat([vert_ids_code, dist, agg_feature], dim=-1) # torch.Size([16384, 669])
        else:
            f = torch.cat([query_pts_code, t_vertex, dist], dim=-1) # torch.Size([16384, 669])
            # f = torch.cat([vert_ids_code, dist], dim=-1) # torch.Size([16384, 669])
        return f
    
    def backward_fusion(self, canonical_pts, vertices, vert_ids, agg_feature=None):
        # (X_c, Pose_s, shape, x-joint_c, f_agg) → deformation Net 2 → X_s
        canonical_pts_code = self.pos_enc(canonical_pts)
        vert_ids = vert_ids.squeeze(0)
        vert_ids_code = self.pos_enc(vert_ids.float())
        dist = self.pdist2(canonical_pts, torch.from_numpy(vertices).cuda().float()[vert_ids].squeeze(1))
        # pts = torch.repeat_interleave(canonical_pts.unsqueeze(1), 24, dim=1) # torch.Size([16384, 24, 3])
        # c_joints = torch.from_numpy(self.c_joints).cuda().float() #  torch.Size([24, 3]
        # diff = torch.reshape(pts - c_joints, [pts.shape[0], -1] ) # N * 72
        # poses = torch.repeat_interleave(torch.from_numpy(params['poses']).cuda().float(), pts.shape[0], 0)
        # shapes = torch.repeat_interleave(torch.from_numpy(params['shapes']).cuda().float(), pts.shape[0], 0)

        if agg_feature != None:
            # f = torch.cat([canonical_pts_code, diff, poses, shapes, agg_feature], dim=-1) # torch.Size([16384, 669])
            # f = torch.cat([canonical_pts_code, diff, agg_feature], dim=-1) # torch.Size([16384, 669])
            f = torch.cat([canonical_pts_code, dist, agg_feature], dim=-1) # torch.Size([16384, 669])
            # f = torch.cat([vert_ids_code, dist, agg_feature], dim=-1) # torch.Size([16384, 669])
        else:
            # f = torch.cat([vert_ids_code, dist], dim=-1) # torch.Size([16384, 669])
            f = torch.cat([canonical_pts_code, dist], dim=-1) # torch.Size([16384, 669])
            # f = torch.cat([canonical_pts_code, diff], dim=-1) # torch.Size([16384, 669])
            # f = torch.cat([canonical_pts_code, diff, poses, shapes], dim=-1) # torch.Size([16384, 669])

        # self.fuse2 = nn.Linear(512, 512, 1)
        return f

    def normalize_pts(self, source_pts, bounds):
        # bounds 2 * 3
        # source_pts N * 3
        mean_plus = 0.5 * (bounds[0] + bounds[1])
        mean_plus = torch.from_numpy(mean_plus).cuda().float()
        mean_sub = 0.5 * (bounds[1] - bounds[0])
        mean_sub = torch.from_numpy(mean_sub).cuda().float()

        normalized_pts = ((source_pts - mean_plus) / mean_sub)

        return normalized_pts

    def forward(self, sp_input, tp_input, query_pts, viewdir):
        # translate query points from world space target pose to smpl space target pose
        # Rh = tp_input['params']['Rh']
        # R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        R = torch.from_numpy(tp_input['params']['R']).cuda().float()
        Th = tp_input['params']['Th'].astype(np.float32)
        Th = torch.from_numpy(Th).cuda().float()
        query_pts = torch.mm(query_pts - Th, R)

        img_all = sp_input['img_all']
        img_all = torch.from_numpy(img_all).cuda().float()
        self.encode_images(img_all)
        
        # query points in smpl space source pose
        coarse_canonical_pts,vert_ids = self.coarse_deform_target2c(tp_input['params'], tp_input['vertices'], query_pts)        
        
        if self.use_agg:
            coarse_smpl_src_pts, coarse_world_src_pts, _ = self.coarse_deform_c2source(sp_input['params'], sp_input['t_vertices'], query_pts=coarse_canonical_pts)
            uv = self.projection(coarse_world_src_pts, sp_input['R_all'], sp_input['T_all'], sp_input['K_all'])
            agg_feature = self.window_feature(uv) # 512 * N
        else:
            agg_feature = None
        # (X_t, pose_t, shape, x-joint_t, f_agg) → deformation Net 1 → X_c
        fused_feature = self.forward_fusion(query_pts, tp_input['vertices'], vert_ids, tp_input['t_vertices'], agg_feature)
        correction = self.forward_deform(fused_feature.float())
        canonical_pts = coarse_canonical_pts + correction # self.forward_transform(self.t_A, skinning_weight, query_pts)

        # (X_c, Pose_s, shape, x-joint_c, f_agg) → deformation Net 2 → X_s
        coarse_smpl_src_pts, _, vert_ids = self.coarse_deform_c2source(sp_input['params'], sp_input['t_vertices'], query_pts=canonical_pts)
        fused_feature = self.backward_fusion(canonical_pts, tp_input['t_vertices'], vert_ids,  agg_feature)
        correction_ = self.backward_deform(fused_feature.float())
        smpl_src_pts = coarse_smpl_src_pts + correction_ # self.backward_transform(self.s_A, skinning_weight, canonical_pts)
    
        # translate source points from smpl space source pose to world space source pose
        Rh = sp_input['params']['Rh']
        R = torch.tensor(cv2.Rodrigues(Rh)[0].astype(np.float32)).cuda().float()
        R_inv = torch.inverse(R)
        Th = torch.tensor(sp_input['params']['Th'].astype(np.float32)).cuda().float()
        world_src_pts = torch.mm(smpl_src_pts, R_inv) + Th
        
        xyz_input_feature = torch.from_numpy(sp_input['t_feature']).cuda().float()
        voxel_integer_coord = torch.from_numpy(sp_input['t_coord']).cuda().float().int()
        coord = torch.zeros_like(voxel_integer_coord[:,0:1])
        voxel_integer_coord = torch.cat((coord, voxel_integer_coord), dim=1)
        out_sh = sp_input['t_out_sh']
        batch_size = 1
        # get geometry aligned feature
        sparse_smpl_vertices = spconv.SparseConvTensor(xyz_input_feature, voxel_integer_coord, out_sh, batch_size)
        normalized_source_pts = self.normalize_pts(canonical_pts, sp_input['t_bounds'])
        point_3d_feature = self.encoder_3d(sparse_smpl_vertices, normalized_source_pts)
        # point_3d_feature = self.encoder_3d(sparse_smpl_vertices, normalized_source_pts[:, [2, 1, 0]])
        
        uv = self.projection(world_src_pts, sp_input['R_all'], sp_input['T_all'], sp_input['K_all'])
        # get mean pixel-aligned feature four view
        point_2d_feature = self.encoder_2d.index(uv, self.image_shape)
        point_2d_feature = point_2d_feature.transpose(1,2)
        point_2d_feature = torch.sum(point_2d_feature, dim=0)


        # mlp_input: (point + f3d + f3d)
        x = self.pos_enc(canonical_pts)
        if self.use_f2d:
            x = torch.cat((x, point_3d_feature, point_2d_feature), dim=1)
        else:
            x = torch.cat((x, point_3d_feature), dim=1)
        
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        
        viewdir = self.view_enc(viewdir)
        # h = torch.cat([feature, viewdir, point_3d_feature, point_2d_feature], -1)
        h = torch.cat([feature, viewdir, point_2d_feature], -1)
    
        h = self.views_linear(h)
        h = F.relu(h)

        rgb = self.rgb_linear(h)
        # raw = torch.cat([rgb, alpha], -1)
        raw = torch.cat([rgb, alpha, query_pts, smpl_src_pts, correction, correction_], -1)
        
        return raw