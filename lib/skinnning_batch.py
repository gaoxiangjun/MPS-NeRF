from random import sample
from lib.run_nerf_helpers import PositionalEncoding, wide_sigmoid
import torch.nn as nn
from pytorch3d.ops.knn import knn_points
import spconv.pytorch as spconv
import torch.nn.functional as F
import torch
import cv2
from lib.encoder import *
from lib.base_utils import read_pickle
# from h36m_blend_weights import *
from lib.transformer import Transformer
from lib.run_nerf_helpers import *
import copy


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = torch.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    norm = torch.zeros(vertices.shape, dtype=vertices.dtype).cuda()
    tris = vertices[faces]
    # n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    n = torch.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    normalize_v3(n)
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat
    

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
            outputs = F.softmax(outputs, dim=1)

        return outputs


class SKinningBatch(nn.Module):
    def __init__(self, use_agg=False, human_sample=True, density_loss=False, with_viewdirs=False, use_f2d=True, use_trans=False, 
                smooth_loss=False, num_instances=1, mean_shape=1, correction_field=1, skinning_field=1, data_set_type="H36M_B", append_rgb="False"):
        super(SKinningBatch, self).__init__()
        self.encoder_3d = SparseConvNet(num_layers=2)
        self.encoder_2d = SpatialEncoder(num_layers=2)# no smooth loss

        self.forward_deform = DeformField(D=2, input_ch=(39+32+128), output_ch=3, deform_type='correction')
        self.backward_deform = DeformField(D=4, input_ch=(39 + 128), output_ch=24, deform_type='weights')

        self.with_viewdirs = with_viewdirs
        self.pos_enc = PositionalEncoding(num_freqs=6)
        self.view_enc = PositionalEncoding(num_freqs=4)
        
        male_smpl_path = os.path.join('assets', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
        female_smpl_path = os.path.join('assets', 'basicmodel_f_lbs_10_207_0_v1.0.0.pkl')
        neutral_smpl_path = os.path.join('assets', 'SMPL_NEUTRAL.pkl')
        self.SMPL_MALE = SMPL_to_tensor(read_pickle(male_smpl_path))
        self.SMPL_FEMALE = SMPL_to_tensor(read_pickle(female_smpl_path))
        self.SMPL_NEU = SMPL_to_tensor(read_pickle(neutral_smpl_path))
        self.SMPL_NEUTRAL = self.SMPL_NEU # SMPL_to_tensor(read_pickle(smpl_path))

        self.faces = self.SMPL_NEUTRAL['f'] #.astype(np.int32)
        self.smpl_pts_normal = None #compute_normal(t_vertices, faces)
        W = 256
        self.image_shape = torch.zeros((2,))

        self.actvn = nn.ReLU()
        self.skips = [4]
        nerf_input_ch = (39+128) if not append_rgb else (39+128+27)#if smooth_loss else (39+32+128)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(nerf_input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + nerf_input_ch, W) for i in range(7)])
        self.alpha_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)
        self.rgb_linear = nn.Linear(W//2, 3)
        
        self.extract_mesh = False
        self.mesh_animation = False
        self.use_agg = use_agg
        self.data_set_type = data_set_type
        self.human_sample = human_sample
        self.density_loss = density_loss
        self.use_f2d = use_f2d
        self.use_trans = use_trans
        self.smooth_loss = smooth_loss
        self.mean_shape = mean_shape
        self.correction_field = correction_field
        self.skinning_field = skinning_field
        self.append_rgb = append_rgb

        nerf_input_ch_2 = 384 if not append_rgb else (128+256+27) # 128 fused feature + 256 alpha feature + rgb code 
        self.views_linear = nn.Linear((nerf_input_ch_2+27) if self.with_viewdirs else nerf_input_ch_2, W//2)
        self.transformer = None if not use_trans else Transformer(128 if not append_rgb else (128+27))
        self.latent_codes = nn.Embedding(num_instances, 128)
        self.extract_mesh = False
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

    def normalize_pts(self, source_pts, bounds):
        # bounds 2 * 3
        # source_pts N * 3
        mean_plus = 0.5 * (bounds[0] + bounds[1])
        mean_plus = mean_plus
        mean_sub = 0.5 * (bounds[1] - bounds[0])
        mean_sub = mean_sub

        normalized_pts = ((source_pts - mean_plus) / mean_sub)
        return normalized_pts

    def projection(self, query_pts, R, T, K):
        RT = torch.cat([R, T], 2)
        xyz = torch.repeat_interleave(query_pts.unsqueeze(dim=0), repeats=RT.shape[0], dim=0)
        xyz = torch.bmm(xyz.float(), RT[:, :, :3].transpose(1, 2).float()) + RT[:, :, 3:].transpose(1, 2).float()
        xyz = torch.bmm(xyz, K.transpose(1, 2).float())
        xy = xyz[:, :, :2] / (xyz[:, :, 2:] + 1e-5)
        
        return xy
    
    def encode_images(self, images):
        self.images = images
        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        self.encoder_2d(images)

    def big_pose_params(self, params):
        big_pose_params = copy.deepcopy(params)
        big_pose_params['poses'] = torch.zeros((1,72))
        big_pose_params['poses'][0, 5] = 45/180*torch.tensor(np.pi)
        big_pose_params['poses'][0, 8] = -45/180*torch.tensor(np.pi)
        big_pose_params['poses'][0, 23] = -30/180*torch.tensor(np.pi)
        big_pose_params['poses'][0, 26] = 30/180*torch.tensor(np.pi)

        return big_pose_params

    def coarse_deform_target2c(self, params, vertices,  query_pts):

        # joints transformation
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params)        
        J = torch.hstack((joints, torch.ones([joints.shape[0], 1])))
        # self.c_joints = joints # smpl space
        # self.t_A = A # target to canonical space transformation
        # self.t_joints = torch.matmul(A, J.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]

        # transform smpl vertices from world space to smpl space
        smpl_pts = torch.mm((vertices - Th), R)
        _, vert_ids, _ = knn_points(query_pts.unsqueeze(0).float(), smpl_pts.unsqueeze(0).float(), K=1)
        bweights = self.SMPL_NEUTRAL['weights'][vert_ids.squeeze(0)].view(-1,24).cuda()

        # From smpl space target pose to smpl space canonical pose
        A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = query_pts - A[:, :3, 3]
        R_inv = torch.inverse(A[:, :3, :3].float())
        can_pts = torch.sum(R_inv * can_pts[:, None], dim=2)
        
        if self.mean_shape:
            posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
            pose_ = params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = 1
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1]).cuda()
            pose_offsets = torch.matmul(pose_feature, posedirs.view(6890*3, -1).transpose(1,0)).view(-1, 3)
            can_pts = can_pts - pose_offsets[vert_ids.squeeze(0).reshape(-1)]

            # To mean shape
            shapedirs = self.SMPL_NEUTRAL['shapedirs'].cuda()
            # shape_offset = shapedirs.dot(torch.reshape(params['shapes'], (10,)))
            # shape_offset = torch.einsum('ijk, k -> ij', shapedirs, torch.reshape(params['shapes'], (10,)))
            shape_offset = (shapedirs * torch.reshape(params['shapes'].cuda(), (10,))).sum(dim=-1)
            can_pts = can_pts - shape_offset[vert_ids.squeeze(0).reshape(-1)]

        # From T To Big Pose        
        big_pose_params = self.big_pose_params(params)
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, big_pose_params)
        # J = torch.hstack((self.c_joints, torch.ones([self.c_joints.shape[0], 1])))
        # self.c_joints = torch.matmul(A, J.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = torch.sum(A[:, :3, :3] * can_pts[:, None], dim=2)
        can_pts = can_pts + A[:, :3, 3]

        return can_pts 

    def coarse_deform_c2source(self, params, t_vertices, query_pts, weights_correction):
        # Find nearest smpl vertex
        smpl_pts = t_vertices
        _, vert_ids, _ = knn_points(query_pts.unsqueeze(0).float(), smpl_pts.unsqueeze(0).float(), K=1)
        bweights = self.SMPL_NEUTRAL['weights'][vert_ids.squeeze(0)].view(-1,24).cuda()

        # add weights_correction, normalize weights
        # bweights = F.softmax(bweights + 0.2*weights_correction, dim=1)
        bweights = bweights + 0.2 * weights_correction
        bweights = bweights / torch.sum(bweights, dim=-1, keepdim=True)

        ### From Big To T Pose
        big_pose_params = self.big_pose_params(params)
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, big_pose_params)
        A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        query_pts = query_pts - A[:, :3, 3]
        R_inv = torch.inverse(A[:, :3, :3].float())
        query_pts = torch.sum(R_inv * query_pts[:, None], dim=2)

        if self.mean_shape:
            # From mean shape to normal shape
            shapedirs = self.SMPL_NEUTRAL['shapedirs'].cuda()
            shape_offset = (shapedirs * torch.reshape(params['shapes'].cuda(), (10,))).sum(dim=-1)
            query_pts = query_pts + shape_offset[vert_ids.squeeze(0).reshape(-1)]

            posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
            pose_ = params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = 1
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1]).cuda()
            pose_offsets = torch.matmul(pose_feature, posedirs.view(6890*3, -1).transpose(1,0)).view(-1, 3)
            query_pts = query_pts + pose_offsets[vert_ids.squeeze(0).reshape(-1)]

        # get tar_pts, smpl space source pose
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params)
        self.s_A = A
        A = torch.mm(bweights, self.s_A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = torch.sum(A[:, :3, :3] * query_pts[:, None], dim=2)
        smpl_src_pts = can_pts + A[:, :3, 3]
        
        # transform points from the smpl space to the world space
        R_inv = torch.inverse(R)
        world_src_pts = torch.mm(smpl_src_pts, R_inv) + Th
        
        return smpl_src_pts, world_src_pts, bweights

    def forward_fusion(self, query_pts, params, point_3d_feature, agg_feature=None):
        """(x, x-joint_t, target_f_3d) → deformation Net 1 → X_c"""
        query_pts_code = self.pos_enc(query_pts)
        if agg_feature != None:
            f = torch.cat([query_pts_code, point_3d_feature, agg_feature], dim=-1) # 39 + 72 + 32 + 128 = 271
        else:
            f = torch.cat([query_pts_code, point_3d_feature], dim=-1) # torch.Size([16384, 669])
        return f
    
    def backward_fusion(self, canonical_pts, embedding=None):
        """(X_c, Pose_s, shape, x-joint_c, f_agg) → deformation Net 2 → X_s"""
        canonical_pts_code = self.pos_enc(canonical_pts) # torch.Size([16384, 3])
        embedding = torch.repeat_interleave(embedding.unsqueeze(0), canonical_pts_code.shape[0], dim=0) # torch.Size([16384, 24, 3])
        f = torch.cat([canonical_pts_code, embedding], dim=-1) # torch.Size([16384, 669])
        return f
    
    def prepare_spconv(self, tp_input, t_vertices=False):
        pre = 't_' if t_vertices else ''
        xyz_input_feature = tp_input[pre+'feature'] #torch.from_numpy(tp_input['feature']).cuda().float()
        voxel_integer_coord = tp_input[pre+'coord'].int() #torch.from_numpy(tp_input['coord']).cuda().float().int()
        coord = torch.zeros_like(voxel_integer_coord[:,0:1])
        voxel_integer_coord = torch.cat((coord, voxel_integer_coord), dim=1)
        out_sh = tp_input[pre+'out_sh'].cpu().numpy().astype(np.int32)
        batch_size = 1
        sparse_smpl_vertices = spconv.SparseConvTensor(xyz_input_feature, voxel_integer_coord, out_sh, batch_size)

        return sparse_smpl_vertices

    def set_extract_mesh(self, flag):
        self.extract_mesh = flag

    def forward(self, sp_input, tp_input, world_query_pts, viewdir):

        if sp_input["gender"][0] == 1:
            self.SMPL_NEUTRAL = self.SMPL_MALE
        elif sp_input["gender"][0] == 0:
            self.SMPL_NEUTRAL = self.SMPL_FEMALE
        else:
            self.SMPL_NEUTRAL = self.SMPL_NEU
        ## translate query points from world space target pose to smpl space target pose
        world_query_pts = world_query_pts.squeeze(0)
        viewdir = viewdir.squeeze(0)
        sp_input, tp_input = sequeeze_0(sp_input, tp_input)
        R = tp_input['params']['R']
        Th = tp_input['params']['Th'] #.astype(np.float32)
        smpl_query_pts = torch.mm(world_query_pts - Th, R)
        
        ## encode images
        img_all = sp_input['img_all']
        self.encode_images(img_all)
        
        ## human region sample
        if self.human_sample:
            tar_smpl_pts = tp_input['vertices']
            tar_smpl_pts = torch.mm(tar_smpl_pts - Th, R)
            distance, _, _ = knn_points(smpl_query_pts.unsqueeze(0).float(), tar_smpl_pts.unsqueeze(0).float(), K=1)
            distance = distance.squeeze(0).view(-1)
            pts_mask = torch.zeros_like(smpl_query_pts[:,0]).cuda().int()
            threshold = 0.05 ** 2
            pts_mask[distance < threshold] = 1
            
            zero_pts = torch.zeros(smpl_query_pts.shape).cuda().float()
            smpl_query_pts = smpl_query_pts[pts_mask==1]
            viewdir = viewdir[pts_mask==1]
        else:
            zero_pts = torch.zeros(smpl_query_pts.shape).cuda().float()
            pts_mask = torch.ones_like(smpl_query_pts[:,0]).cuda().int()
        
        ## coarse deform target to caonical
        coarse_canonical_pts = self.coarse_deform_target2c(tp_input['params'], tp_input['vertices'], smpl_query_pts)        
        
        ## Calculate correction
        if self.correction_field or self.skinning_field:
            sparse_smpl_vertices = self.prepare_spconv(tp_input, t_vertices=False)
            normalized_source_pts = self.normalize_pts(smpl_query_pts, tp_input['bounds'])
            point_3d_feature_0 = self.encoder_3d(sparse_smpl_vertices, normalized_source_pts.detach())
            if self.data_set_type in ["H36M_P", "THuman_P"]:
                _, coarse_world_src_pts, _ = self.coarse_deform_c2source(sp_input['params'], sp_input['t_vertices'], query_pts=coarse_canonical_pts, weights_correction=0.)
                uv = self.projection(coarse_world_src_pts, sp_input['R_all'], sp_input['T_all'], sp_input['K_all'])
            else:
                uv = self.projection(world_query_pts[pts_mask==1].detach(), sp_input['R_all'], sp_input['T_all'], sp_input['K_all'])
            agg_feature = torch.mean(self.encoder_2d.index(uv, self.image_shape), dim=0).transpose(0,1)        
            fused_feature = self.forward_fusion(smpl_query_pts, tp_input['params'], point_3d_feature_0, agg_feature)
        
        ## More Accurate canonical_pts
        if self.correction_field:
            correction = self.forward_deform(fused_feature.float())
            canonical_pts = coarse_canonical_pts + correction
        else:
            canonical_pts = coarse_canonical_pts.requires_grad_()

        ## Extract Mesh
        if self.extract_mesh or self.mesh_animation:
            canonical_pts = world_query_pts
            pts_mask = torch.ones_like(canonical_pts[:,0]).cuda().int()

        ## Use Skinning weights Field to do deform
        instance_idx = tp_input['instance_idx'].long()
        embedding = self.latent_codes(instance_idx)
        fused_feature = self.backward_fusion(canonical_pts.clone(), embedding)
        if self.skinning_field:
            weights_correction = self.backward_deform(fused_feature.float())
        else:
            weights_correction = 0.
        smpl_src_pts, world_src_pts, bweights = self.coarse_deform_c2source(sp_input['params'], sp_input['t_vertices'], canonical_pts, weights_correction)
        
        if self.smooth_loss and self.training:
            _, vert_ids, _ = knn_points(canonical_pts.unsqueeze(0).float(), sp_input['t_vertices'].unsqueeze(0).float(), K=1)
            if self.smpl_pts_normal==None:
                self.smpl_pts_normal = compute_normal(sp_input['t_vertices'], self.faces)
            nearest_smpl_normal = self.smpl_pts_normal[vert_ids.squeeze(0)].reshape(-1, 3)

        ## Get canonical 3d geometry aligned feature
        # sparse_smpl_vertices = self.prepare_spconv(tp_input, t_vertices=True)
        # normalized_source_pts = self.normalize_pts(canonical_pts, sp_input['t_bounds'])
        # point_3d_feature = self.encoder_3d(sparse_smpl_vertices, normalized_source_pts)
        
        ## Get mean pixel-aligned feature four view
        uv = self.projection(world_src_pts, sp_input['R_all'], sp_input['T_all'], sp_input['K_all'])
        # if self.correction_field or tp_input['pose_index']!=sp_input['pose_index']:
        #     uv = self.projection(world_src_pts, sp_input['R_all'], sp_input['T_all'], sp_input['K_all'])
        # else:
        #     uv = self.projection(world_query_pts[pts_mask==1], sp_input['R_all'], sp_input['T_all'], sp_input['K_all'])
        point_2d_feature_0 = self.encoder_2d.index(uv, self.image_shape)
        point_2d_feature_0 = point_2d_feature_0.transpose(1,2)
        
        if self.append_rgb:
            img = sp_input['img_all']
            uv_ = 2.0 * uv.unsqueeze(2).type(torch.float32)  / self.image_shape.clone().detach().to(uv.device) - 1.0
            samples = grid_sample(img, uv_)
            samples = samples.squeeze(-1).transpose(1,2)
            sh = samples.shape
            samples = self.view_enc(samples.reshape(-1,3)).reshape(sh[0],sh[1],27)
            point_2d_feature_0 = torch.cat((point_2d_feature_0, samples), dim=-1)

        ## Transformer Fusion of multiview Feature
        if self.use_trans and point_2d_feature_0.shape[1]!=0:
            point_2d_feature = point_2d_feature_0.transpose(0,1)
            point_2d_feature = self.transformer(point_2d_feature)
            point_2d_feature_1 = point_2d_feature[:,0,:]
            point_2d_feature_2 = point_2d_feature[:,1,:]
        else:
            point_2d_feature = torch.mean(point_2d_feature_0, dim=0)
            point_2d_feature_1 = point_2d_feature
            point_2d_feature_2 = point_2d_feature

        # Run in Nerf
        x = self.pos_enc(canonical_pts)
        # if not self.smooth_loss:
        #     x = torch.cat((x, point_3d_feature, point_2d_feature_1), dim=1)
        # else:
        #     x = torch.cat((x, point_2d_feature_1), dim=1)
        x = torch.cat((x, point_2d_feature_1), dim=1)
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
            h = torch.cat([feature, viewdir, point_2d_feature_2], -1)
        else:
            h = torch.cat([feature, point_2d_feature_2], -1)

        h = self.views_linear(h)
        h = F.relu(h)
        rgb = self.rgb_linear(h)
        
        if self.mesh_animation:
            raw = torch.cat([world_src_pts, bweights, rgb, alpha], dim=-1)
            return raw
        if self.extract_mesh:
            raw = torch.cat([rgb, alpha], -1).unsqueeze(0)
            return raw

        if self.human_sample:
            ret_smpl_query_pts, ret_smpl_src_pts = torch.zeros(zero_pts.shape).cuda().float(), torch.zeros(zero_pts.shape).cuda().float()
            ret_smpl_query_pts[pts_mask==1], ret_smpl_src_pts[pts_mask==1] = smpl_query_pts, smpl_src_pts
            ret_correction, ret_correction_ = torch.zeros(zero_pts.shape).cuda().float(), torch.zeros(zero_pts.shape).cuda().float()
            all_occ_normal, all_nearest_smpl_normal = torch.zeros(zero_pts.shape).cuda().float(), torch.zeros(zero_pts.shape).cuda().float()
            
            if self.correction_field:
                ret_correction[pts_mask==1] = correction

            raw = torch.zeros(zero_pts.shape[0],4).cuda().float()
            raw[pts_mask==1] = torch.cat([rgb, alpha], -1)
            raw[pts_mask==0] = -80 # softplus and sigmoid is outside
            raw = torch.cat([raw, torch.unsqueeze(pts_mask, -1), ret_correction, ret_correction_, ret_smpl_query_pts, ret_smpl_src_pts], -1)

            if self.smooth_loss and self.training:
                intv_flag = (int(sp_input['global_step'].item()) % int(sp_input['smooth_interval'].item()))==0
                if intv_flag:
                    occ_normal = torch.autograd.grad(wide_sigmoid(alpha), [canonical_pts], grad_outputs=torch.ones_like(alpha), create_graph=True)[0]
                    occ_normal = occ_normal / (torch.norm(occ_normal, dim=-1, keepdim=True) + 1e-8)
                    all_occ_normal[pts_mask==1] = occ_normal
                    all_nearest_smpl_normal[pts_mask==1] = nearest_smpl_normal
                    # raw = torch.cat([raw, all_occ_normal], -1)
                    raw = torch.cat([raw, all_occ_normal, all_nearest_smpl_normal], -1)

                    del x, h, point_2d_feature, point_2d_feature_0, weights_correction, fused_feature
                    del feature, canonical_pts, smpl_src_pts, world_src_pts, instance_idx
                    del coarse_canonical_pts, pts_mask, sp_input, tp_input, world_query_pts, viewdir, occ_normal
        else:
            pts_mask = torch.ones_like(smpl_query_pts[:,0]).cuda().int()
            pts_mask = torch.unsqueeze(pts_mask, -1)
            raw = torch.cat([rgb, alpha, pts_mask, correction], -1)
        
        return raw.unsqueeze(0)


class DirectDeform(nn.Module):
    def __init__(self, use_agg=False, human_sample=True, density_loss=False, with_viewdirs=False, use_f2d=True, use_trans=False, 
                smooth_loss=False, num_instances=1, mean_shape=1, correction_field=1, skinning_field=1, data_set_type="H36M_B", append_rgb="False"):
        super(DirectDeform, self).__init__()
        self.encoder_3d = SparseConvNet(num_layers=2)
        self.encoder_2d = SpatialEncoder(num_layers=2)# no smooth loss
        # self.encoder_2d = ImageViewEncoder(num_layers=2)# smooth loss
        self.forward_deform = DeformField(D=2, input_ch=(39+32+128), output_ch=3, deform_type='correction')
        self.backward_deform = DeformField(D=4, input_ch=(39 + 128), output_ch=24, deform_type='weights')
        self.use_agg = use_agg
        self.with_viewdirs = with_viewdirs
        self.pos_enc = PositionalEncoding(num_freqs=6)
        self.view_enc = PositionalEncoding(num_freqs=4)
        
        male_smpl_path = os.path.join('assets', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
        female_smpl_path = os.path.join('assets', 'basicmodel_f_lbs_10_207_0_v1.0.0.pkl')
        neutral_smpl_path = os.path.join('assets', 'SMPL_NEUTRAL.pkl')
        self.SMPL_MALE = SMPL_to_tensor(read_pickle(male_smpl_path))
        self.SMPL_FEMALE = SMPL_to_tensor(read_pickle(female_smpl_path))
        self.SMPL_NEU = SMPL_to_tensor(read_pickle(neutral_smpl_path))

        smpl_path = os.path.join('assets', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl') if data_set_type[0]=="T" else os.path.join('msra_h36m', 'smplx/smpl/SMPL_NEUTRAL.pkl')
        self.data_set_type = data_set_type
        self.SMPL_NEUTRAL = self.SMPL_NEU # SMPL_to_tensor(read_pickle(smpl_path))
        # self.SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(os.path.join('msra_h36m', 'smplx/smpl/SMPL_NEUTRAL.pkl')))
        # self.SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(os.path.join('assets', 'basicmodel_f_lbs_10_207_0_v1.0.0.pkl')))
        self.faces = self.SMPL_NEUTRAL['f'] #.astype(np.int32)
        self.smpl_pts_normal = None #compute_normal(t_vertices, faces)
        W = 256
        self.image_shape = torch.zeros((2,))

        # xc + f3d + f2d
        self.actvn = nn.ReLU()
        self.skips = [4]
        nerf_input_ch = (128) if not append_rgb else (128+27)#if smooth_loss else (39+32+128)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(nerf_input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + nerf_input_ch, W) for i in range(7)])
        self.alpha_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)
        self.rgb_linear = nn.Linear(W//2, 3)
        self.extract_mesh = False
        self.mesh_animation = False
        self.human_sample = human_sample
        self.density_loss = density_loss
        self.use_f2d = use_f2d
        self.use_trans = use_trans
        self.smooth_loss = smooth_loss
        self.mean_shape = mean_shape
        self.correction_field = correction_field
        self.skinning_field = skinning_field
        self.append_rgb = append_rgb

        nerf_input_ch_2 = 384 if not append_rgb else (128+256+27) # 128 fused feature + 256 alpha feature + rgb code 
        self.views_linear = nn.Linear((nerf_input_ch_2+27) if self.with_viewdirs else nerf_input_ch_2, W//2)
        self.transformer = None if not use_trans else Transformer(128 if not append_rgb else (128+27))
        self.latent_codes = nn.Embedding(num_instances, 128)
        self.extract_mesh = False
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

    def normalize_pts(self, source_pts, bounds):
        # bounds 2 * 3
        # source_pts N * 3
        mean_plus = 0.5 * (bounds[0] + bounds[1])
        mean_plus = mean_plus
        mean_sub = 0.5 * (bounds[1] - bounds[0])
        mean_sub = mean_sub

        normalized_pts = ((source_pts - mean_plus) / mean_sub)
        return normalized_pts

    def projection(self, query_pts, R, T, K):
        # https://github.com/pengsida/if-nerf/blob/492e765423fcbfb9aeca345286a49074cc4e5a90/lib/utils/render_utils.py#L45
        RT = torch.cat([R, T], 2)
        xyz = torch.repeat_interleave(query_pts.unsqueeze(dim=0), repeats=RT.shape[0], dim=0)
        xyz = torch.bmm(xyz.float(), RT[:, :, :3].transpose(1, 2).float()) + RT[:, :, 3:].transpose(1, 2).float()
        xyz = torch.bmm(xyz, K.transpose(1, 2).float())
        xy = xyz[:, :, :2] / (xyz[:, :, 2:] + 1e-5)
        
        return xy
    
    def encode_images(self, images):
        self.images = images
        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        self.encoder_2d(images)

    def big_pose_params(self, params):
        big_pose_params = copy.deepcopy(params)
        big_pose_params['poses'] = torch.zeros((1,72))
        big_pose_params['poses'][0, 5] = 45/180*torch.tensor(np.pi)
        big_pose_params['poses'][0, 8] = -45/180*torch.tensor(np.pi)
        big_pose_params['poses'][0, 23] = -30/180*torch.tensor(np.pi)
        big_pose_params['poses'][0, 26] = 30/180*torch.tensor(np.pi)

        return big_pose_params

    def coarse_deform_target2c(self, params, vertices,  query_pts):

        # joints transformation
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params)        
        J = torch.hstack((joints, torch.ones([joints.shape[0], 1])))
        # self.c_joints = joints # smpl space
        # self.t_A = A # target to canonical space transformation
        # self.t_joints = torch.matmul(A, J.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]

        # transform smpl vertices from world space to smpl space
        smpl_pts = torch.mm((vertices - Th), R)
        _, vert_ids, _ = knn_points(query_pts.unsqueeze(0).float(), smpl_pts.unsqueeze(0).float(), K=1)
        bweights = self.SMPL_NEUTRAL['weights'][vert_ids.squeeze(0)].view(-1,24).cuda()

        # From smpl space target pose to smpl space canonical pose
        A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = query_pts - A[:, :3, 3]
        R_inv = torch.inverse(A[:, :3, :3].float())
        can_pts = torch.sum(R_inv * can_pts[:, None], dim=2)
        
        if self.mean_shape:
            posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
            pose_ = params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = 1
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1]).cuda()
            pose_offsets = torch.matmul(pose_feature, posedirs.view(6890*3, -1).transpose(1,0)).view(-1, 3)
            can_pts = can_pts - pose_offsets[vert_ids.squeeze(0).reshape(-1)]

            # To mean shape
            shapedirs = self.SMPL_NEUTRAL['shapedirs'].cuda()
            # shape_offset = shapedirs.dot(torch.reshape(params['shapes'], (10,)))
            # shape_offset = torch.einsum('ijk, k -> ij', shapedirs, torch.reshape(params['shapes'], (10,)))
            shape_offset = (shapedirs * torch.reshape(params['shapes'].cuda(), (10,))).sum(dim=-1)
            can_pts = can_pts - shape_offset[vert_ids.squeeze(0).reshape(-1)]

        # From T To Big Pose        
        big_pose_params = self.big_pose_params(params)
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, big_pose_params)
        # J = torch.hstack((self.c_joints, torch.ones([self.c_joints.shape[0], 1])))
        # self.c_joints = torch.matmul(A, J.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = torch.sum(A[:, :3, :3] * can_pts[:, None], dim=2)
        can_pts = can_pts + A[:, :3, 3]

        return can_pts 

    def coarse_deform_c2source(self, params, t_vertices, query_pts, weights_correction):
        # Find nearest smpl vertex
        smpl_pts = t_vertices
        _, vert_ids, _ = knn_points(query_pts.unsqueeze(0).float(), smpl_pts.unsqueeze(0).float(), K=1)
        bweights = self.SMPL_NEUTRAL['weights'][vert_ids.squeeze(0)].view(-1,24).cuda()

        # add weights_correction, normalize weights
        # bweights = F.softmax(bweights + 0.2*weights_correction, dim=1)
        bweights = bweights + 0.2 * weights_correction
        bweights = bweights / torch.sum(bweights, dim=-1, keepdim=True)

        ### From Big To T Pose
        big_pose_params = self.big_pose_params(params)
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, big_pose_params)
        A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        query_pts = query_pts - A[:, :3, 3]
        R_inv = torch.inverse(A[:, :3, :3].float())
        query_pts = torch.sum(R_inv * query_pts[:, None], dim=2)

        if self.mean_shape:
            # From mean shape to normal shape
            shapedirs = self.SMPL_NEUTRAL['shapedirs'].cuda()
            shape_offset = (shapedirs * torch.reshape(params['shapes'].cuda(), (10,))).sum(dim=-1)
            query_pts = query_pts + shape_offset[vert_ids.squeeze(0).reshape(-1)]

            posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
            pose_ = params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = 1
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1]).cuda()
            pose_offsets = torch.matmul(pose_feature, posedirs.view(6890*3, -1).transpose(1,0)).view(-1, 3)
            query_pts = query_pts + pose_offsets[vert_ids.squeeze(0).reshape(-1)]

        # get tar_pts, smpl space source pose
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params)
        self.s_A = A
        A = torch.mm(bweights, self.s_A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = torch.sum(A[:, :3, :3] * query_pts[:, None], dim=2)
        smpl_src_pts = can_pts + A[:, :3, 3]
        
        # transform points from the smpl space to the world space
        R_inv = torch.inverse(R)
        world_src_pts = torch.mm(smpl_src_pts, R_inv) + Th
        
        return smpl_src_pts, world_src_pts, bweights

    def forward_fusion(self, query_pts, params, point_3d_feature, agg_feature=None):
        """(x, x-joint_t, target_f_3d) → deformation Net 1 → X_c"""
        query_pts_code = self.pos_enc(query_pts)
        if agg_feature != None:
            f = torch.cat([query_pts_code, point_3d_feature, agg_feature], dim=-1) # 39 + 72 + 32 + 128 = 271
        else:
            f = torch.cat([query_pts_code, point_3d_feature], dim=-1) # torch.Size([16384, 669])
        return f
    
    def backward_fusion(self, canonical_pts, embedding=None):
        """(X_c, Pose_s, shape, x-joint_c, f_agg) → deformation Net 2 → X_s"""
        canonical_pts_code = self.pos_enc(canonical_pts) # torch.Size([16384, 3])
        embedding = torch.repeat_interleave(embedding.unsqueeze(0), canonical_pts_code.shape[0], dim=0) # torch.Size([16384, 24, 3])
        f = torch.cat([canonical_pts_code, embedding], dim=-1) # torch.Size([16384, 669])
        return f
    
    def prepare_spconv(self, tp_input, t_vertices=False):
        pre = 't_' if t_vertices else ''
        xyz_input_feature = tp_input[pre+'feature'] #torch.from_numpy(tp_input['feature']).cuda().float()
        voxel_integer_coord = tp_input[pre+'coord'].int() #torch.from_numpy(tp_input['coord']).cuda().float().int()
        coord = torch.zeros_like(voxel_integer_coord[:,0:1])
        voxel_integer_coord = torch.cat((coord, voxel_integer_coord), dim=1)
        out_sh = tp_input[pre+'out_sh'].cpu().numpy().astype(np.int32)
        batch_size = 1
        sparse_smpl_vertices = spconv.SparseConvTensor(xyz_input_feature, voxel_integer_coord, out_sh, batch_size)

        return sparse_smpl_vertices

    def set_extract_mesh(self, flag):
        self.extract_mesh = flag

    def forward(self, sp_input, tp_input, world_query_pts, viewdir):

        if sp_input["gender"][0] == 1:
            self.SMPL_NEUTRAL = self.SMPL_MALE
        elif sp_input["gender"][0] == 0:
            self.SMPL_NEUTRAL = self.SMPL_FEMALE
        else:
            self.SMPL_NEUTRAL = self.SMPL_NEU
        ## translate query points from world space target pose to smpl space target pose
        world_query_pts = world_query_pts.squeeze(0)
        viewdir = viewdir.squeeze(0)
        sp_input, tp_input = sequeeze_0(sp_input, tp_input)
        R = tp_input['params']['R']
        Th = tp_input['params']['Th'] #.astype(np.float32)
        smpl_query_pts = torch.mm(world_query_pts - Th, R)
        
        ## encode images
        img_all = sp_input['img_all']
        self.encode_images(img_all)
        
        ## human region sample
        if self.human_sample:
            tar_smpl_pts = tp_input['vertices']
            tar_smpl_pts = torch.mm(tar_smpl_pts - Th, R)
            distance, _, _ = knn_points(smpl_query_pts.unsqueeze(0).float(), tar_smpl_pts.unsqueeze(0).float(), K=1)
            distance = distance.squeeze(0).view(-1)
            pts_mask = torch.zeros_like(smpl_query_pts[:,0]).cuda().int()
            threshold = 0.05 ** 2
            pts_mask[distance < threshold] = 1
            
            zero_pts = torch.zeros(smpl_query_pts.shape).cuda().float()
            smpl_query_pts = smpl_query_pts[pts_mask==1]
            viewdir = viewdir[pts_mask==1]
        else:
            zero_pts = torch.zeros(smpl_query_pts.shape).cuda().float()
            pts_mask = torch.ones_like(smpl_query_pts[:,0]).cuda().int()
        
        ## coarse deform target to caonical
        coarse_canonical_pts = self.coarse_deform_target2c(tp_input['params'], tp_input['vertices'], smpl_query_pts)        
        canonical_pts = coarse_canonical_pts #.requires_grad_()
        

        ## Extract Mesh
        if self.extract_mesh or self.mesh_animation:
            canonical_pts = world_query_pts
            pts_mask = torch.ones_like(canonical_pts[:,0]).cuda().int()

        ## Use Skinning weights Field to do deform
        weights_correction = 0.
        smpl_src_pts, world_src_pts, bweights = self.coarse_deform_c2source(sp_input['params'], sp_input['t_vertices'], canonical_pts, weights_correction)
        
        if self.smooth_loss and self.training:
            _, vert_ids, _ = knn_points(canonical_pts.unsqueeze(0).float(), sp_input['t_vertices'].unsqueeze(0).float(), K=1)
            if self.smpl_pts_normal==None:
                self.smpl_pts_normal = compute_normal(sp_input['t_vertices'], self.faces)
            nearest_smpl_normal = self.smpl_pts_normal[vert_ids.squeeze(0)].reshape(-1, 3)
        
        uv = self.projection(world_src_pts, sp_input['R_all'], sp_input['T_all'], sp_input['K_all'])
        point_2d_feature_0 = self.encoder_2d.index(uv, self.image_shape)
        point_2d_feature_0 = point_2d_feature_0.transpose(1,2)
        
        if self.append_rgb:
            img = sp_input['img_all']
            uv_ = 2.0 * uv.unsqueeze(2).type(torch.float32)  / self.image_shape.clone().detach().to(uv.device) - 1.0
            samples = grid_sample(img, uv_)
            samples = samples.squeeze(-1).transpose(1,2)
            sh = samples.shape
            samples = self.view_enc(samples.reshape(-1,3)).reshape(sh[0],sh[1],27)
            point_2d_feature_0 = torch.cat((point_2d_feature_0, samples), dim=-1)

        ## Transformer Fusion of multiview Feature
        if self.use_trans and point_2d_feature_0.shape[1]!=0:
            point_2d_feature = point_2d_feature_0.transpose(0,1)
            point_2d_feature = self.transformer(point_2d_feature)
            point_2d_feature_1 = point_2d_feature[:,0,:]
            point_2d_feature_2 = point_2d_feature[:,1,:]
        else:
            point_2d_feature = torch.mean(point_2d_feature_0, dim=0)
            point_2d_feature_1 = point_2d_feature
            point_2d_feature_2 = point_2d_feature

        # Run in Nerf
        # x = self.pos_enc(canonical_pts)
        # x = torch.cat((x, point_2d_feature_1), dim=1)
        x = point_2d_feature_1
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
            h = torch.cat([feature, viewdir, point_2d_feature_2], -1)
        else:
            h = torch.cat([feature, point_2d_feature_2], -1)

        h = self.views_linear(h)
        h = F.relu(h)
        rgb = self.rgb_linear(h)
        
        if self.mesh_animation:
            raw = torch.cat([world_src_pts, bweights, rgb, alpha], dim=-1)
            return raw
        if self.extract_mesh:
            raw = torch.cat([rgb, alpha], -1).unsqueeze(0)
            return raw

        if self.human_sample:
            ret_smpl_query_pts, ret_smpl_src_pts = torch.zeros(zero_pts.shape).cuda().float(), torch.zeros(zero_pts.shape).cuda().float()
            ret_smpl_query_pts[pts_mask==1], ret_smpl_src_pts[pts_mask==1] = smpl_query_pts, smpl_src_pts
            ret_correction, ret_correction_ = torch.zeros(zero_pts.shape).cuda().float(), torch.zeros(zero_pts.shape).cuda().float()
            all_occ_normal, all_nearest_smpl_normal = torch.zeros(zero_pts.shape).cuda().float(), torch.zeros(zero_pts.shape).cuda().float()
            raw = torch.zeros(zero_pts.shape[0],4).cuda().float()
            raw[pts_mask==1] = torch.cat([rgb, alpha], -1)
            raw[pts_mask==0] = -80 # softplus and sigmoid is outside
            raw = torch.cat([raw, torch.unsqueeze(pts_mask, -1), ret_correction, ret_correction_, ret_smpl_query_pts, ret_smpl_src_pts], -1)

            if self.smooth_loss and self.training:
                intv_flag = (int(sp_input['global_step'].item()) % int(sp_input['smooth_interval'].item()))==0
                if intv_flag:
                    occ_normal = torch.autograd.grad(wide_sigmoid(alpha), [canonical_pts], grad_outputs=torch.ones_like(alpha), create_graph=True)[0]
                    occ_normal = occ_normal / (torch.norm(occ_normal, dim=-1, keepdim=True) + 1e-8)
                    all_occ_normal[pts_mask==1] = occ_normal
                    all_nearest_smpl_normal[pts_mask==1] = nearest_smpl_normal
                    # raw = torch.cat([raw, all_occ_normal], -1)
                    raw = torch.cat([raw, all_occ_normal, all_nearest_smpl_normal], -1)
        else:
            pts_mask = torch.ones_like(smpl_query_pts[:,0]).cuda().int()
            pts_mask = torch.unsqueeze(pts_mask, -1)
            raw = torch.cat([rgb, alpha, pts_mask], -1)
        
        return raw.unsqueeze(0)


class CorrectionBatch(nn.Module):
    def __init__(self, use_agg=False, human_sample=True, density_loss=False, with_viewdirs=False, use_f2d=True, use_trans=False, 
                smooth_loss=False, num_instances=1, mean_shape=1, correction_field=1, skinning_field=1, data_set_type="H36M_B", append_rgb="False"):
        super(CorrectionBatch, self).__init__()
        self.encoder_3d = SparseConvNet(num_layers=2)
        self.encoder_2d = SpatialEncoder(num_layers=2)# no smooth loss
        # self.encoder_2d = ImageViewEncoder(num_layers=2)# smooth loss
        self.forward_deform = DeformField(D=4, input_ch=(271), output_ch=3, deform_type='correction')
        self.backward_deform = DeformField(D=4, input_ch=(271), output_ch=3, deform_type='correction')
        # self.forward_deform = DeformField(D=4, input_ch=(39 + 128), output_ch=24, deform_type='weights')
        # self.backward_deform = DeformField(D=4, input_ch=(39 + 128), output_ch=24, deform_type='weights')
        self.use_agg = use_agg
        self.with_viewdirs = with_viewdirs
        self.pos_enc = PositionalEncoding(num_freqs=6)
        self.view_enc = PositionalEncoding(num_freqs=4)

        male_smpl_path = os.path.join('assets', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
        female_smpl_path = os.path.join('assets', 'basicmodel_f_lbs_10_207_0_v1.0.0.pkl')
        neutral_smpl_path = os.path.join('assets', 'SMPL_NEUTRAL.pkl')
        self.SMPL_MALE = SMPL_to_tensor(read_pickle(male_smpl_path))
        self.SMPL_FEMALE = SMPL_to_tensor(read_pickle(female_smpl_path))
        self.SMPL_NEU = SMPL_to_tensor(read_pickle(neutral_smpl_path))

        smpl_path = os.path.join('assets', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl') if data_set_type[0]=="T" else os.path.join('msra_h36m', 'smplx/smpl/SMPL_NEUTRAL.pkl')
        self.data_set_type = data_set_type
        self.SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(smpl_path))
        # self.SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(os.path.join('msra_h36m', 'smplx/smpl/SMPL_NEUTRAL.pkl')))
        # self.SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(os.path.join('assets', 'basicmodel_f_lbs_10_207_0_v1.0.0.pkl')))
        self.faces = self.SMPL_NEUTRAL['f'] #.astype(np.int32)
        self.smpl_pts_normal = None #compute_normal(t_vertices, faces)
        W = 256
        self.image_shape = torch.zeros((2,))

        # xc + f3d + f2d
        self.actvn = nn.ReLU()
        self.skips = [4]
        nerf_input_ch = (39+128) if not append_rgb else (39+128+27)#if smooth_loss else (39+32+128)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(nerf_input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + nerf_input_ch, W) for i in range(7)])
        self.alpha_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)
        self.rgb_linear = nn.Linear(W//2, 3)
        self.extract_mesh = False
        self.mesh_animation = False
        self.human_sample = human_sample
        self.density_loss = density_loss
        self.use_f2d = use_f2d
        self.use_trans = use_trans
        self.smooth_loss = smooth_loss
        self.mean_shape = mean_shape
        self.correction_field = correction_field
        self.skinning_field = skinning_field
        self.append_rgb = append_rgb

        nerf_input_ch_2 = 384 if not append_rgb else (128+256+27) # 128 fused feature + 256 alpha feature + rgb code 
        self.views_linear = nn.Linear((nerf_input_ch_2+27) if self.with_viewdirs else nerf_input_ch_2, W//2)
        self.transformer = None if not use_trans else Transformer(128 if not append_rgb else (128+27))
        self.latent_codes = nn.Embedding(num_instances, 128)
        self.extract_mesh = False
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

    def normalize_pts(self, source_pts, bounds):
        # bounds 2 * 3
        # source_pts N * 3
        mean_plus = 0.5 * (bounds[0] + bounds[1])
        mean_plus = mean_plus
        mean_sub = 0.5 * (bounds[1] - bounds[0])
        mean_sub = mean_sub

        normalized_pts = ((source_pts - mean_plus) / mean_sub)
        return normalized_pts

    def projection(self, query_pts, R, T, K):
        # https://github.com/pengsida/if-nerf/blob/492e765423fcbfb9aeca345286a49074cc4e5a90/lib/utils/render_utils.py#L45
        RT = torch.cat([R, T], 2)
        xyz = torch.repeat_interleave(query_pts.unsqueeze(dim=0), repeats=RT.shape[0], dim=0)
        xyz = torch.bmm(xyz.float(), RT[:, :, :3].transpose(1, 2).float()) + RT[:, :, 3:].transpose(1, 2).float()
        xyz = torch.bmm(xyz, K.transpose(1, 2).float())
        xy = xyz[:, :, :2] / (xyz[:, :, 2:] + 1e-5)
        
        return xy
    
    def encode_images(self, images):
        self.images = images
        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        self.encoder_2d(images)

    def big_pose_params(self, params):
        big_pose_params = copy.deepcopy(params)
        big_pose_params['poses'] = torch.zeros((1,72))
        big_pose_params['poses'][0, 5] = 45/180*torch.tensor(np.pi)
        big_pose_params['poses'][0, 8] = -45/180*torch.tensor(np.pi)
        big_pose_params['poses'][0, 23] = -30/180*torch.tensor(np.pi)
        big_pose_params['poses'][0, 26] = 30/180*torch.tensor(np.pi)

        return big_pose_params

    def coarse_deform_target2c(self, params, vertices,  query_pts):

        # joints transformation
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params)        
        J = torch.hstack((joints, torch.ones([joints.shape[0], 1])))
        # self.c_joints = joints # smpl space
        # self.t_A = A # target to canonical space transformation
        # self.t_joints = torch.matmul(A, J.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]

        # transform smpl vertices from world space to smpl space
        smpl_pts = torch.mm((vertices - Th), R)
        _, vert_ids, _ = knn_points(query_pts.unsqueeze(0).float(), smpl_pts.unsqueeze(0).float(), K=1)
        bweights = self.SMPL_NEUTRAL['weights'][vert_ids.squeeze(0)].view(-1,24).cuda()

        # From smpl space target pose to smpl space canonical pose
        A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = query_pts - A[:, :3, 3]
        R_inv = torch.inverse(A[:, :3, :3].float())
        can_pts = torch.sum(R_inv * can_pts[:, None], dim=2)
        
        if self.mean_shape:
            posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
            pose_ = params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = 1
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1]).cuda()
            pose_offsets = torch.matmul(pose_feature, posedirs.view(6890*3, -1).transpose(1,0)).view(-1, 3)
            can_pts = can_pts - pose_offsets[vert_ids.squeeze(0).reshape(-1)]

            # To mean shape
            shapedirs = self.SMPL_NEUTRAL['shapedirs'].cuda()
            # shape_offset = shapedirs.dot(torch.reshape(params['shapes'], (10,)))
            # shape_offset = torch.einsum('ijk, k -> ij', shapedirs, torch.reshape(params['shapes'], (10,)))
            shape_offset = (shapedirs * torch.reshape(params['shapes'].cuda(), (10,))).sum(dim=-1)
            can_pts = can_pts - shape_offset[vert_ids.squeeze(0).reshape(-1)]

        # From T To Big Pose        
        big_pose_params = self.big_pose_params(params)
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, big_pose_params)
        # J = torch.hstack((self.c_joints, torch.ones([self.c_joints.shape[0], 1])))
        # self.c_joints = torch.matmul(A, J.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = torch.sum(A[:, :3, :3] * can_pts[:, None], dim=2)
        can_pts = can_pts + A[:, :3, 3]

        return can_pts 

    def coarse_deform_c2source(self, params, t_vertices, query_pts, weights_correction):
        # Find nearest smpl vertex
        smpl_pts = t_vertices
        _, vert_ids, _ = knn_points(query_pts.unsqueeze(0).float(), smpl_pts.unsqueeze(0).float(), K=1)
        bweights = self.SMPL_NEUTRAL['weights'][vert_ids.squeeze(0)].view(-1,24).cuda()

        # add weights_correction, normalize weights
        # bweights = F.softmax(bweights + 0.2*weights_correction, dim=1)
        bweights = bweights + 0.2 * weights_correction
        bweights = bweights / torch.sum(bweights, dim=-1, keepdim=True)

        ### From Big To T Pose
        big_pose_params = self.big_pose_params(params)
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, big_pose_params)
        A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        query_pts = query_pts - A[:, :3, 3]
        R_inv = torch.inverse(A[:, :3, :3].float())
        query_pts = torch.sum(R_inv * query_pts[:, None], dim=2)

        if self.mean_shape:
            # From mean shape to normal shape
            shapedirs = self.SMPL_NEUTRAL['shapedirs'].cuda()
            shape_offset = (shapedirs * torch.reshape(params['shapes'].cuda(), (10,))).sum(dim=-1)
            query_pts = query_pts + shape_offset[vert_ids.squeeze(0).reshape(-1)]

            posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
            pose_ = params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = 1
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1]).cuda()
            pose_offsets = torch.matmul(pose_feature, posedirs.view(6890*3, -1).transpose(1,0)).view(-1, 3)
            query_pts = query_pts + pose_offsets[vert_ids.squeeze(0).reshape(-1)]

        # get tar_pts, smpl space source pose
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params)
        self.s_A = A
        A = torch.mm(bweights, self.s_A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = torch.sum(A[:, :3, :3] * query_pts[:, None], dim=2)
        smpl_src_pts = can_pts + A[:, :3, 3]
        
        # transform points from the smpl space to the world space
        R_inv = torch.inverse(R)
        world_src_pts = torch.mm(smpl_src_pts, R_inv) + Th
        
        return smpl_src_pts, world_src_pts, bweights

    def forward_fusion(self, query_pts, params, embedding, point_3d_feature, agg_feature=None):
        """(x, x-joint_t, target_f_3d) → deformation Net 1 → X_c""" #39+72+128+32
        query_pts_code = self.pos_enc(query_pts)
        poses = torch.repeat_interleave(params['poses'].reshape(1,72), query_pts_code.shape[0], 0)
        embedding = torch.repeat_interleave(embedding.unsqueeze(0), query_pts_code.shape[0], 0)
        if agg_feature != None:
            f = torch.cat([query_pts_code, poses, embedding, point_3d_feature, agg_feature], dim=-1) # 39 + 72 + 32 + 128 = 271
        else:
            f = torch.cat([query_pts_code, poses, embedding, point_3d_feature], dim=-1) # torch.Size([16384, 669])
        return f
    
    def backward_fusion(self, query_pts, params, embedding, point_3d_feature):
        """(X_c, Pose_s, shape, x-joint_c, f_agg) → deformation Net 2 → X_s"""
        query_pts_code = self.pos_enc(query_pts) # torch.Size([16384, 3])
        poses = torch.repeat_interleave(params['poses'].reshape(1,72), query_pts_code.shape[0], 0)
        embedding = torch.repeat_interleave(embedding.unsqueeze(0), query_pts_code.shape[0], dim=0) # torch.Size([16384, 24, 3])
        f = torch.cat([query_pts_code, poses, embedding, point_3d_feature], dim=-1) # torch.Size([16384, 669])
        return f
    
    def prepare_spconv(self, tp_input, t_vertices=False):
        pre = 't_' if t_vertices else ''
        xyz_input_feature = tp_input[pre+'feature'] #torch.from_numpy(tp_input['feature']).cuda().float()
        voxel_integer_coord = tp_input[pre+'coord'].int() #torch.from_numpy(tp_input['coord']).cuda().float().int()
        coord = torch.zeros_like(voxel_integer_coord[:,0:1])
        voxel_integer_coord = torch.cat((coord, voxel_integer_coord), dim=1)
        out_sh = tp_input[pre+'out_sh'].cpu().numpy().astype(np.int32)
        batch_size = 1
        sparse_smpl_vertices = spconv.SparseConvTensor(xyz_input_feature, voxel_integer_coord, out_sh, batch_size)

        return sparse_smpl_vertices

    def set_extract_mesh(self, flag):
        self.extract_mesh = flag

    def forward(self, sp_input, tp_input, world_query_pts, viewdir):

        ## translate query points from world space target pose to smpl space target pose
        world_query_pts = world_query_pts.squeeze(0)
        viewdir = viewdir.squeeze(0)
        sp_input, tp_input = sequeeze_0(sp_input, tp_input)
        R = tp_input['params']['R']
        Th = tp_input['params']['Th'] #.astype(np.float32)
        smpl_query_pts = torch.mm(world_query_pts - Th, R)
        
        if sp_input["gender"] == "M":
            self.SMPL_NEUTRAL = self.SMPL_MALE
        elif sp_input["gender"] == "F":
            self.SMPL_NEUTRAL = self.SMPL_FEMALE
        else:
            self.SMPL_NEUTRAL = self.SMPL_NEU

        ## encode images
        img_all = sp_input['img_all']
        self.encode_images(img_all)
        
        ## human region sample
        if self.human_sample:
            tar_smpl_pts = tp_input['vertices']
            tar_smpl_pts = torch.mm(tar_smpl_pts - Th, R)
            distance, _, _ = knn_points(smpl_query_pts.unsqueeze(0).float(), tar_smpl_pts.unsqueeze(0).float(), K=1)
            distance = distance.squeeze(0).view(-1)
            pts_mask = torch.zeros_like(smpl_query_pts[:,0]).cuda().int()
            threshold = 0.05 ** 2
            pts_mask[distance < threshold] = 1
            
            zero_pts = torch.zeros(smpl_query_pts.shape).cuda().float()
            smpl_query_pts = smpl_query_pts[pts_mask==1]
            viewdir = viewdir[pts_mask==1]
        else:
            zero_pts = torch.zeros(smpl_query_pts.shape).cuda().float()
            pts_mask = torch.ones_like(smpl_query_pts[:,0]).cuda().int()
        
        ## More Accurate canonical_pts
        if self.correction_field:
            sparse_smpl_vertices = self.prepare_spconv(tp_input, t_vertices=False)
            normalized_source_pts = self.normalize_pts(smpl_query_pts, tp_input['bounds'])
            point_3d_feature_0 = self.encoder_3d(sparse_smpl_vertices, normalized_source_pts.detach())
            
            instance_idx = tp_input['instance_idx'].long()
            embedding = self.latent_codes(instance_idx)
            fused_feature = self.forward_fusion(smpl_query_pts, tp_input['params'], embedding, point_3d_feature_0)
            
            correction = self.forward_deform(fused_feature.float())
            # canonical_pts = self.coarse_deform_target2c(tp_input['params'], tp_input['vertices'], smpl_query_pts + correction)
            canonical_pts = self.coarse_deform_target2c(tp_input['params'], tp_input['vertices'], smpl_query_pts)
            canonical_pts = canonical_pts + correction
        else:
            canonical_pts = self.coarse_deform_target2c(tp_input['params'], tp_input['vertices'], smpl_query_pts).requires_grad_()

        ## Extract Mesh
        if self.extract_mesh or self.mesh_animation:
            canonical_pts = world_query_pts
            pts_mask = torch.ones_like(canonical_pts[:,0]).cuda().int()

        smpl_src_pts, world_src_pts, bweights = self.coarse_deform_c2source(sp_input['params'], sp_input['t_vertices'], canonical_pts, weights_correction=0.)
        
        if self.correction_field:
            sparse_smpl_vertices = self.prepare_spconv(sp_input, t_vertices=False)
            normalized_source_pts = self.normalize_pts(smpl_src_pts, sp_input['bounds'])
            point_3d_feature_0 = self.encoder_3d(sparse_smpl_vertices, normalized_source_pts.detach())
            
            instance_idx = sp_input['instance_idx'].long()
            embedding = self.latent_codes(instance_idx)
            fused_feature = self.backward_fusion(smpl_src_pts.clone(), sp_input['params'], embedding, point_3d_feature_0)
            
            correction_ = self.backward_deform(fused_feature.float())
            smpl_src_pts = smpl_src_pts + correction_
            A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, sp_input['params'])
            R_inv = torch.inverse(R)
            world_src_pts = torch.mm(smpl_src_pts, R_inv) + Th

        if self.smooth_loss and self.training:
            _, vert_ids, _ = knn_points(canonical_pts.unsqueeze(0).float(), sp_input['t_vertices'].unsqueeze(0).float(), K=1)
            if self.smpl_pts_normal==None:
                self.smpl_pts_normal = compute_normal(sp_input['t_vertices'], self.faces)
            nearest_smpl_normal = self.smpl_pts_normal[vert_ids.squeeze(0)].reshape(-1, 3)

        ## Get multi-view pixel-aligned feature four view
        if self.correction_field or tp_input['pose_index']!=sp_input['pose_index']:
            uv = self.projection(world_src_pts, sp_input['R_all'], sp_input['T_all'], sp_input['K_all'])
        else:
            uv = self.projection(world_query_pts[pts_mask==1], sp_input['R_all'], sp_input['T_all'], sp_input['K_all'])
        point_2d_feature_0 = self.encoder_2d.index(uv, self.image_shape)
        point_2d_feature_0 = point_2d_feature_0.transpose(1,2)
        
        if self.append_rgb:
            img = sp_input['img_all']
            uv_ = 2.0 * uv.unsqueeze(2).type(torch.float32)  / self.image_shape.clone().detach().to(uv.device) - 1.0
            samples = grid_sample(img, uv_)
            samples = samples.squeeze(-1).transpose(1,2)
            sh = samples.shape
            samples = self.view_enc(samples.reshape(-1,3)).reshape(sh[0],sh[1],27)
            point_2d_feature_0 = torch.cat((point_2d_feature_0, samples), dim=-1)

        ## Transformer Fusion of multiview Feature
        if self.use_trans and point_2d_feature_0.shape[1]!=0:
            point_2d_feature = point_2d_feature_0.transpose(0,1)
            point_2d_feature = self.transformer(point_2d_feature)
            point_2d_feature_1 = point_2d_feature[:,0,:]
            point_2d_feature_2 = point_2d_feature[:,1,:]
        else:
            point_2d_feature = torch.mean(point_2d_feature_0, dim=0)
            point_2d_feature_1 = point_2d_feature
            point_2d_feature_2 = point_2d_feature

        # Run in Nerf
        x = self.pos_enc(canonical_pts)
        x = torch.cat((x, point_2d_feature_1), dim=1)
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
            h = torch.cat([feature, viewdir, point_2d_feature_2], -1)
        else:
            h = torch.cat([feature, point_2d_feature_2], -1)

        h = self.views_linear(h)
        h = F.relu(h)
        rgb = self.rgb_linear(h)
        
        if self.mesh_animation:
            raw = torch.cat([world_src_pts, bweights, rgb, alpha], dim=-1)
            return raw
        if self.extract_mesh:
            raw = torch.cat([rgb, alpha], -1).unsqueeze(0)
            return raw

        if self.human_sample:
            ret_smpl_query_pts, ret_smpl_src_pts = torch.zeros(zero_pts.shape).cuda().float(), torch.zeros(zero_pts.shape).cuda().float()
            ret_smpl_query_pts[pts_mask==1], ret_smpl_src_pts[pts_mask==1] = smpl_query_pts, smpl_src_pts
            ret_correction, ret_correction_ = torch.zeros(zero_pts.shape).cuda().float(), torch.zeros(zero_pts.shape).cuda().float()
            all_occ_normal, all_nearest_smpl_normal = torch.zeros(zero_pts.shape).cuda().float(), torch.zeros(zero_pts.shape).cuda().float()
            if self.correction_field:
                ret_correction[pts_mask==1] = correction
                ret_correction_[pts_mask==1] = correction_

            raw = torch.zeros(zero_pts.shape[0],4).cuda().float()
            raw[pts_mask==1] = torch.cat([rgb, alpha], -1)
            raw[pts_mask==0] = -80 # softplus and sigmoid is outside
            raw = torch.cat([raw, torch.unsqueeze(pts_mask, -1), ret_correction, ret_correction_, ret_smpl_query_pts, ret_smpl_src_pts], -1)

            if self.smooth_loss and self.training:
                intv_flag = (int(sp_input['global_step'].item()) % int(sp_input['smooth_interval'].item()))==0
                if intv_flag:
                    # TODO wide_sigmoid(alpha)
                    # occ_normal = torch.autograd.grad(alpha, [canonical_pts], grad_outputs=torch.ones_like(alpha), create_graph=True)[0]
                    occ_normal = torch.autograd.grad(wide_sigmoid(alpha), [canonical_pts], grad_outputs=torch.ones_like(alpha), create_graph=True)[0]
                    occ_normal = occ_normal / (torch.norm(occ_normal, dim=-1, keepdim=True) + 1e-8)
                    all_occ_normal[pts_mask==1] = occ_normal
                    all_nearest_smpl_normal[pts_mask==1] = nearest_smpl_normal
                    raw = torch.cat([raw, all_occ_normal, all_nearest_smpl_normal], -1)

                    del x, h, point_2d_feature, point_2d_feature_0, fused_feature
                    del feature, point_3d_feature_0, canonical_pts
                    del pts_mask, sp_input, tp_input, world_query_pts, viewdir, occ_normal
        else:
            pts_mask = torch.ones_like(smpl_query_pts[:,0]).cuda().int()
            pts_mask = torch.unsqueeze(pts_mask, -1)
            raw = torch.cat([rgb, alpha, pts_mask, correction], -1)
        
        return raw.unsqueeze(0)


class CoarseBatch(nn.Module):
    def __init__(self, use_agg=False, human_sample=True, density_loss=False, with_viewdirs=False, use_f2d=True, use_trans=False, 
                smooth_loss=False, num_instances=1, mean_shape=1, correction_field=1, skinning_field=1, data_set_type="H36M_B", append_rgb="False"):
        super(CorrectionBatch, self).__init__()
        # self.encoder_3d = SparseConvNet(num_layers=2)
        self.encoder_2d = SpatialEncoder(num_layers=2)# no smooth loss
        # self.encoder_2d = ImageViewEncoder(num_layers=2)# smooth loss
        # self.forward_deform = DeformField(D=4, input_ch=(271), output_ch=3, deform_type='correction')
        # self.backward_deform = DeformField(D=4, input_ch=(271), output_ch=3, deform_type='correction')
        # self.forward_deform = DeformField(D=4, input_ch=(39 + 128), output_ch=24, deform_type='weights')
        # self.backward_deform = DeformField(D=4, input_ch=(39 + 128), output_ch=24, deform_type='weights')
        self.use_agg = use_agg
        self.with_viewdirs = with_viewdirs
        self.pos_enc = PositionalEncoding(num_freqs=6)
        self.view_enc = PositionalEncoding(num_freqs=4)

        male_smpl_path = os.path.join('assets', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
        female_smpl_path = os.path.join('assets', 'basicmodel_f_lbs_10_207_0_v1.0.0.pkl')
        neutral_smpl_path = os.path.join('assets', 'SMPL_NEUTRAL.pkl')
        self.SMPL_MALE = SMPL_to_tensor(read_pickle(male_smpl_path))
        self.SMPL_FEMALE = SMPL_to_tensor(read_pickle(female_smpl_path))
        self.SMPL_NEU = SMPL_to_tensor(read_pickle(neutral_smpl_path))

        smpl_path = os.path.join('assets', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl') if data_set_type[0]=="T" else os.path.join('msra_h36m', 'smplx/smpl/SMPL_NEUTRAL.pkl')
        self.data_set_type = data_set_type
        self.SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(smpl_path))
        # self.SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(os.path.join('msra_h36m', 'smplx/smpl/SMPL_NEUTRAL.pkl')))
        # self.SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(os.path.join('assets', 'basicmodel_f_lbs_10_207_0_v1.0.0.pkl')))
        self.faces = self.SMPL_NEUTRAL['f'] #.astype(np.int32)
        self.smpl_pts_normal = None #compute_normal(t_vertices, faces)
        W = 256
        self.image_shape = torch.zeros((2,))

        # xc + f3d + f2d
        self.actvn = nn.ReLU()
        self.skips = [4]
        nerf_input_ch = (39+128) if not append_rgb else (39+128+27)#if smooth_loss else (39+32+128)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(nerf_input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + nerf_input_ch, W) for i in range(7)])
        self.alpha_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)
        self.rgb_linear = nn.Linear(W//2, 3)
        self.extract_mesh = False
        self.mesh_animation = False
        self.human_sample = human_sample
        self.density_loss = density_loss
        self.use_f2d = use_f2d
        self.use_trans = use_trans
        self.smooth_loss = smooth_loss
        self.mean_shape = mean_shape
        self.correction_field = correction_field
        self.skinning_field = skinning_field
        self.append_rgb = append_rgb

        nerf_input_ch_2 = 384 if not append_rgb else (128+256+27) # 128 fused feature + 256 alpha feature + rgb code 
        self.views_linear = nn.Linear((nerf_input_ch_2+27) if self.with_viewdirs else nerf_input_ch_2, W//2)
        self.transformer = None if not use_trans else Transformer(128 if not append_rgb else (128+27))
        self.latent_codes = nn.Embedding(num_instances, 128)
        self.extract_mesh = False
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

    def normalize_pts(self, source_pts, bounds):
        # bounds 2 * 3
        # source_pts N * 3
        mean_plus = 0.5 * (bounds[0] + bounds[1])
        mean_plus = mean_plus
        mean_sub = 0.5 * (bounds[1] - bounds[0])
        mean_sub = mean_sub

        normalized_pts = ((source_pts - mean_plus) / mean_sub)
        return normalized_pts

    def projection(self, query_pts, R, T, K):
        # https://github.com/pengsida/if-nerf/blob/492e765423fcbfb9aeca345286a49074cc4e5a90/lib/utils/render_utils.py#L45
        RT = torch.cat([R, T], 2)
        xyz = torch.repeat_interleave(query_pts.unsqueeze(dim=0), repeats=RT.shape[0], dim=0)
        xyz = torch.bmm(xyz.float(), RT[:, :, :3].transpose(1, 2).float()) + RT[:, :, 3:].transpose(1, 2).float()
        xyz = torch.bmm(xyz, K.transpose(1, 2).float())
        xy = xyz[:, :, :2] / (xyz[:, :, 2:] + 1e-5)
        
        return xy
    
    def encode_images(self, images):
        self.images = images
        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        self.encoder_2d(images)

    def big_pose_params(self, params):
        big_pose_params = copy.deepcopy(params)
        big_pose_params['poses'] = torch.zeros((1,72))
        big_pose_params['poses'][0, 5] = 45/180*torch.tensor(np.pi)
        big_pose_params['poses'][0, 8] = -45/180*torch.tensor(np.pi)
        big_pose_params['poses'][0, 23] = -30/180*torch.tensor(np.pi)
        big_pose_params['poses'][0, 26] = 30/180*torch.tensor(np.pi)

        return big_pose_params

    def coarse_deform_target2c(self, params, vertices,  query_pts):

        # joints transformation
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params)        
        J = torch.hstack((joints, torch.ones([joints.shape[0], 1])))
        # self.c_joints = joints # smpl space
        # self.t_A = A # target to canonical space transformation
        # self.t_joints = torch.matmul(A, J.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]

        # transform smpl vertices from world space to smpl space
        smpl_pts = torch.mm((vertices - Th), R)
        _, vert_ids, _ = knn_points(query_pts.unsqueeze(0).float(), smpl_pts.unsqueeze(0).float(), K=1)
        bweights = self.SMPL_NEUTRAL['weights'][vert_ids.squeeze(0)].view(-1,24).cuda()

        # From smpl space target pose to smpl space canonical pose
        A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = query_pts - A[:, :3, 3]
        R_inv = torch.inverse(A[:, :3, :3].float())
        can_pts = torch.sum(R_inv * can_pts[:, None], dim=2)
        
        if self.mean_shape:
            posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
            pose_ = params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = 1
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1]).cuda()
            pose_offsets = torch.matmul(pose_feature, posedirs.view(6890*3, -1).transpose(1,0)).view(-1, 3)
            can_pts = can_pts - pose_offsets[vert_ids.squeeze(0).reshape(-1)]

            # To mean shape
            shapedirs = self.SMPL_NEUTRAL['shapedirs'].cuda()
            # shape_offset = shapedirs.dot(torch.reshape(params['shapes'], (10,)))
            # shape_offset = torch.einsum('ijk, k -> ij', shapedirs, torch.reshape(params['shapes'], (10,)))
            shape_offset = (shapedirs * torch.reshape(params['shapes'].cuda(), (10,))).sum(dim=-1)
            can_pts = can_pts - shape_offset[vert_ids.squeeze(0).reshape(-1)]

        # From T To Big Pose        
        big_pose_params = self.big_pose_params(params)
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, big_pose_params)
        # J = torch.hstack((self.c_joints, torch.ones([self.c_joints.shape[0], 1])))
        # self.c_joints = torch.matmul(A, J.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = torch.sum(A[:, :3, :3] * can_pts[:, None], dim=2)
        can_pts = can_pts + A[:, :3, 3]

        return can_pts 

    def coarse_deform_c2source(self, params, t_vertices, query_pts, weights_correction):
        # Find nearest smpl vertex
        smpl_pts = t_vertices
        _, vert_ids, _ = knn_points(query_pts.unsqueeze(0).float(), smpl_pts.unsqueeze(0).float(), K=1)
        bweights = self.SMPL_NEUTRAL['weights'][vert_ids.squeeze(0)].view(-1,24).cuda()

        # add weights_correction, normalize weights
        # bweights = F.softmax(bweights + 0.2*weights_correction, dim=1)
        bweights = bweights + 0.2 * weights_correction
        bweights = bweights / torch.sum(bweights, dim=-1, keepdim=True)

        ### From Big To T Pose
        big_pose_params = self.big_pose_params(params)
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, big_pose_params)
        A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        query_pts = query_pts - A[:, :3, 3]
        R_inv = torch.inverse(A[:, :3, :3].float())
        query_pts = torch.sum(R_inv * query_pts[:, None], dim=2)

        if self.mean_shape:
            # From mean shape to normal shape
            shapedirs = self.SMPL_NEUTRAL['shapedirs'].cuda()
            shape_offset = (shapedirs * torch.reshape(params['shapes'].cuda(), (10,))).sum(dim=-1)
            query_pts = query_pts + shape_offset[vert_ids.squeeze(0).reshape(-1)]

            posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
            pose_ = params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = 1
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1]).cuda()
            pose_offsets = torch.matmul(pose_feature, posedirs.view(6890*3, -1).transpose(1,0)).view(-1, 3)
            query_pts = query_pts + pose_offsets[vert_ids.squeeze(0).reshape(-1)]

        # get tar_pts, smpl space source pose
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params)
        self.s_A = A
        A = torch.mm(bweights, self.s_A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = torch.sum(A[:, :3, :3] * query_pts[:, None], dim=2)
        smpl_src_pts = can_pts + A[:, :3, 3]
        
        # transform points from the smpl space to the world space
        R_inv = torch.inverse(R)
        world_src_pts = torch.mm(smpl_src_pts, R_inv) + Th
        
        return smpl_src_pts, world_src_pts, bweights

    def set_extract_mesh(self, flag):
        self.extract_mesh = flag

    def forward(self, sp_input, tp_input, world_query_pts, viewdir):

        ## translate query points from world space target pose to smpl space target pose
        world_query_pts = world_query_pts.squeeze(0)
        viewdir = viewdir.squeeze(0)
        sp_input, tp_input = sequeeze_0(sp_input, tp_input)
        R = tp_input['params']['R']
        Th = tp_input['params']['Th'] #.astype(np.float32)
        smpl_query_pts = torch.mm(world_query_pts - Th, R)
        
        if sp_input["gender"] == "M":
            self.SMPL_NEUTRAL = self.SMPL_MALE
        elif sp_input["gender"] == "F":
            self.SMPL_NEUTRAL = self.SMPL_FEMALE
        else:
            self.SMPL_NEUTRAL = self.SMPL_NEU

        ## encode images
        img_all = sp_input['img_all']
        self.encode_images(img_all)
        
        ## human region sample
        if self.human_sample:
            tar_smpl_pts = tp_input['vertices']
            tar_smpl_pts = torch.mm(tar_smpl_pts - Th, R)
            distance, _, _ = knn_points(smpl_query_pts.unsqueeze(0).float(), tar_smpl_pts.unsqueeze(0).float(), K=1)
            distance = distance.squeeze(0).view(-1)
            pts_mask = torch.zeros_like(smpl_query_pts[:,0]).cuda().int()
            threshold = 0.05 ** 2
            pts_mask[distance < threshold] = 1
            
            zero_pts = torch.zeros(smpl_query_pts.shape).cuda().float()
            smpl_query_pts = smpl_query_pts[pts_mask==1]
            viewdir = viewdir[pts_mask==1]
        else:
            zero_pts = torch.zeros(smpl_query_pts.shape).cuda().float()
            pts_mask = torch.ones_like(smpl_query_pts[:,0]).cuda().int()
        
        ## More Accurate canonical_pts
        canonical_pts = self.coarse_deform_target2c(tp_input['params'], tp_input['vertices'], smpl_query_pts).requires_grad_()

        ## Extract Mesh
        if self.extract_mesh or self.mesh_animation:
            canonical_pts = world_query_pts
            pts_mask = torch.ones_like(canonical_pts[:,0]).cuda().int()

        smpl_src_pts, world_src_pts, bweights = self.coarse_deform_c2source(sp_input['params'], sp_input['t_vertices'], canonical_pts, weights_correction=0.)

        if self.smooth_loss and self.training:
            _, vert_ids, _ = knn_points(canonical_pts.unsqueeze(0).float(), sp_input['t_vertices'].unsqueeze(0).float(), K=1)
            if self.smpl_pts_normal==None:
                self.smpl_pts_normal = compute_normal(sp_input['t_vertices'], self.faces)
            nearest_smpl_normal = self.smpl_pts_normal[vert_ids.squeeze(0)].reshape(-1, 3)

        ## Get multi-view pixel-aligned feature four view
        if self.correction_field or tp_input['pose_index']!=sp_input['pose_index']:
            uv = self.projection(world_src_pts, sp_input['R_all'], sp_input['T_all'], sp_input['K_all'])
        else:
            uv = self.projection(world_query_pts[pts_mask==1], sp_input['R_all'], sp_input['T_all'], sp_input['K_all'])
        point_2d_feature_0 = self.encoder_2d.index(uv, self.image_shape)
        point_2d_feature_0 = point_2d_feature_0.transpose(1,2)
        
        if self.append_rgb:
            img = sp_input['img_all']
            uv_ = 2.0 * uv.unsqueeze(2).type(torch.float32)  / self.image_shape.clone().detach().to(uv.device) - 1.0
            samples = grid_sample(img, uv_)
            samples = samples.squeeze(-1).transpose(1,2)
            sh = samples.shape
            samples = self.view_enc(samples.reshape(-1,3)).reshape(sh[0],sh[1],27)
            point_2d_feature_0 = torch.cat((point_2d_feature_0, samples), dim=-1)

        ## Transformer Fusion of multiview Feature
        if self.use_trans and point_2d_feature_0.shape[1]!=0:
            point_2d_feature = point_2d_feature_0.transpose(0,1)
            point_2d_feature = self.transformer(point_2d_feature)
            point_2d_feature_1 = point_2d_feature[:,0,:]
            point_2d_feature_2 = point_2d_feature[:,1,:]
        else:
            point_2d_feature = torch.mean(point_2d_feature_0, dim=0)
            point_2d_feature_1 = point_2d_feature
            point_2d_feature_2 = point_2d_feature

        # Run in Nerf
        x = self.pos_enc(canonical_pts)
        x = torch.cat((x, point_2d_feature_1), dim=1)
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
            h = torch.cat([feature, viewdir, point_2d_feature_2], -1)
        else:
            h = torch.cat([feature, point_2d_feature_2], -1)

        h = self.views_linear(h)
        h = F.relu(h)
        rgb = self.rgb_linear(h)
        
        if self.mesh_animation:
            raw = torch.cat([world_src_pts, bweights, rgb, alpha], dim=-1)
            return raw
        if self.extract_mesh:
            raw = torch.cat([rgb, alpha], -1).unsqueeze(0)
            return raw

        if self.human_sample:
            ret_smpl_query_pts, ret_smpl_src_pts = torch.zeros(zero_pts.shape).cuda().float(), torch.zeros(zero_pts.shape).cuda().float()
            ret_smpl_query_pts[pts_mask==1], ret_smpl_src_pts[pts_mask==1] = smpl_query_pts, smpl_src_pts
            ret_correction, ret_correction_ = torch.zeros(zero_pts.shape).cuda().float(), torch.zeros(zero_pts.shape).cuda().float()
            all_occ_normal, all_nearest_smpl_normal = torch.zeros(zero_pts.shape).cuda().float(), torch.zeros(zero_pts.shape).cuda().float()

            raw = torch.zeros(zero_pts.shape[0],4).cuda().float()
            raw[pts_mask==1] = torch.cat([rgb, alpha], -1)
            raw[pts_mask==0] = -80 # softplus and sigmoid is outside
            raw = torch.cat([raw, torch.unsqueeze(pts_mask, -1), ret_correction, ret_correction_, ret_smpl_query_pts, ret_smpl_src_pts], -1)

            if self.smooth_loss and self.training:
                intv_flag = (int(sp_input['global_step'].item()) % int(sp_input['smooth_interval'].item()))==0
                if intv_flag:
                    # TODO wide_sigmoid(alpha)
                    # occ_normal = torch.autograd.grad(alpha, [canonical_pts], grad_outputs=torch.ones_like(alpha), create_graph=True)[0]
                    occ_normal = torch.autograd.grad(wide_sigmoid(alpha), [canonical_pts], grad_outputs=torch.ones_like(alpha), create_graph=True)[0]
                    occ_normal = occ_normal / (torch.norm(occ_normal, dim=-1, keepdim=True) + 1e-8)
                    all_occ_normal[pts_mask==1] = occ_normal
                    all_nearest_smpl_normal[pts_mask==1] = nearest_smpl_normal
                    raw = torch.cat([raw, all_occ_normal, all_nearest_smpl_normal], -1)

                    del x, h, point_2d_feature, feature, canonical_pts, point_2d_feature_0
                    del pts_mask, sp_input, tp_input, world_query_pts, viewdir, occ_normal
        else:
            pts_mask = torch.ones_like(smpl_query_pts[:,0]).cuda().int()
            pts_mask = torch.unsqueeze(pts_mask, -1)
            raw = torch.cat([rgb, alpha, pts_mask, correction], -1)
        
        return raw.unsqueeze(0)


class CorrectionBatchBlend(nn.Module):
    def __init__(self, use_agg=False, human_sample=True, density_loss=False, with_viewdirs=False, use_f2d=True, use_trans=False, 
                smooth_loss=False, num_instances=1, mean_shape=1, correction_field=1, skinning_field=1, data_set_type="H36M_B", append_rgb="False"):
        super(CorrectionBatchBlend, self).__init__()
        self.encoder_3d = SparseConvNet(num_layers=2)
        self.encoder_2d = SpatialEncoder(num_layers=2)# no smooth loss
        # self.encoder_2d = ImageViewEncoder(num_layers=2)# smooth loss
        self.forward_deform = DeformField(D=4, input_ch=(271), output_ch=3, deform_type='correction')
        self.backward_deform = DeformField(D=4, input_ch=(271), output_ch=3, deform_type='correction')
        # self.forward_deform = DeformField(D=4, input_ch=(39 + 128), output_ch=24, deform_type='weights')
        # self.backward_deform = DeformField(D=4, input_ch=(39 + 128), output_ch=24, deform_type='weights')
        self.use_agg = use_agg
        self.with_viewdirs = with_viewdirs
        self.pos_enc = PositionalEncoding(num_freqs=6)
        self.view_enc = PositionalEncoding(num_freqs=4)

        male_smpl_path = os.path.join('assets', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
        female_smpl_path = os.path.join('assets', 'basicmodel_f_lbs_10_207_0_v1.0.0.pkl')
        neutral_smpl_path = os.path.join('assets', 'SMPL_NEUTRAL.pkl')
        self.SMPL_MALE = SMPL_to_tensor(read_pickle(male_smpl_path))
        self.SMPL_FEMALE = SMPL_to_tensor(read_pickle(female_smpl_path))
        self.SMPL_NEU = SMPL_to_tensor(read_pickle(neutral_smpl_path))

        smpl_path = os.path.join('assets', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl') if data_set_type[0]=="T" else os.path.join('msra_h36m', 'smplx/smpl/SMPL_NEUTRAL.pkl')
        self.data_set_type = data_set_type
        self.SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(smpl_path))

        self.faces = self.SMPL_NEUTRAL['f'] #.astype(np.int32)
        self.smpl_pts_normal = None #compute_normal(t_vertices, faces)
        W = 256
        self.image_shape = torch.zeros((2,))

        # xc + f3d + f2d
        self.actvn = nn.ReLU()
        self.skips = [4]
        nerf_input_ch = (39+128) if not append_rgb else (39+128+27)#if smooth_loss else (39+32+128)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(nerf_input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + nerf_input_ch, W) for i in range(7)])
        self.alpha_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)
        self.rgb_linear = nn.Linear(W//2, 4)
        self.extract_mesh = False
        self.mesh_animation = False
        self.human_sample = human_sample
        self.density_loss = density_loss
        self.use_f2d = use_f2d
        self.use_trans = use_trans
        self.smooth_loss = smooth_loss
        self.mean_shape = mean_shape
        self.correction_field = correction_field
        self.skinning_field = skinning_field
        self.append_rgb = append_rgb

        nerf_input_ch_2 = 384 if not append_rgb else (128+256+27) # 128 fused feature + 256 alpha feature + rgb code 
        self.views_linear = nn.Linear((nerf_input_ch_2+27) if self.with_viewdirs else nerf_input_ch_2, W//2)
        self.transformer = None if not use_trans else Transformer(128 if not append_rgb else (128+27))
        self.latent_codes = nn.Embedding(num_instances, 128)
        self.extract_mesh = False
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

    def normalize_pts(self, source_pts, bounds):
        # bounds 2 * 3
        # source_pts N * 3
        mean_plus = 0.5 * (bounds[0] + bounds[1])
        mean_plus = mean_plus
        mean_sub = 0.5 * (bounds[1] - bounds[0])
        mean_sub = mean_sub

        normalized_pts = ((source_pts - mean_plus) / mean_sub)
        return normalized_pts

    def projection(self, query_pts, R, T, K):
        # https://github.com/pengsida/if-nerf/blob/492e765423fcbfb9aeca345286a49074cc4e5a90/lib/utils/render_utils.py#L45
        RT = torch.cat([R, T], 2)
        xyz = torch.repeat_interleave(query_pts.unsqueeze(dim=0), repeats=RT.shape[0], dim=0)
        xyz = torch.bmm(xyz.float(), RT[:, :, :3].transpose(1, 2).float()) + RT[:, :, 3:].transpose(1, 2).float()
        xyz = torch.bmm(xyz, K.transpose(1, 2).float())
        xy = xyz[:, :, :2] / (xyz[:, :, 2:] + 1e-5)
        
        return xy
    
    def encode_images(self, images):
        self.images = images
        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        self.encoder_2d(images)

    def big_pose_params(self, params):
        big_pose_params = copy.deepcopy(params)
        big_pose_params['poses'] = torch.zeros((1,72))
        big_pose_params['poses'][0, 5] = 45/180*torch.tensor(np.pi)
        big_pose_params['poses'][0, 8] = -45/180*torch.tensor(np.pi)
        big_pose_params['poses'][0, 23] = -30/180*torch.tensor(np.pi)
        big_pose_params['poses'][0, 26] = 30/180*torch.tensor(np.pi)

        return big_pose_params

    def coarse_deform_target2c(self, params, vertices,  query_pts):

        # joints transformation
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params)        
        J = torch.hstack((joints, torch.ones([joints.shape[0], 1])))
        # self.c_joints = joints # smpl space
        # self.t_A = A # target to canonical space transformation
        # self.t_joints = torch.matmul(A, J.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]

        # transform smpl vertices from world space to smpl space
        smpl_pts = torch.mm((vertices - Th), R)
        _, vert_ids, _ = knn_points(query_pts.unsqueeze(0).float(), smpl_pts.unsqueeze(0).float(), K=1)
        bweights = self.SMPL_NEUTRAL['weights'][vert_ids.squeeze(0)].view(-1,24).cuda()

        # From smpl space target pose to smpl space canonical pose
        A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = query_pts - A[:, :3, 3]
        R_inv = torch.inverse(A[:, :3, :3].float())
        can_pts = torch.sum(R_inv * can_pts[:, None], dim=2)
        
        if self.mean_shape:
            posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
            pose_ = params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = 1
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1]).cuda()
            pose_offsets = torch.matmul(pose_feature, posedirs.view(6890*3, -1).transpose(1,0)).view(-1, 3)
            can_pts = can_pts - pose_offsets[vert_ids.squeeze(0).reshape(-1)]

            # To mean shape
            shapedirs = self.SMPL_NEUTRAL['shapedirs'].cuda()
            # shape_offset = shapedirs.dot(torch.reshape(params['shapes'], (10,)))
            # shape_offset = torch.einsum('ijk, k -> ij', shapedirs, torch.reshape(params['shapes'], (10,)))
            shape_offset = (shapedirs * torch.reshape(params['shapes'].cuda(), (10,))).sum(dim=-1)
            can_pts = can_pts - shape_offset[vert_ids.squeeze(0).reshape(-1)]

        # From T To Big Pose        
        big_pose_params = self.big_pose_params(params)
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, big_pose_params)
        # J = torch.hstack((self.c_joints, torch.ones([self.c_joints.shape[0], 1])))
        # self.c_joints = torch.matmul(A, J.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = torch.sum(A[:, :3, :3] * can_pts[:, None], dim=2)
        can_pts = can_pts + A[:, :3, 3]

        return can_pts 

    def coarse_deform_c2source(self, params, t_vertices, query_pts, weights_correction):
        # Find nearest smpl vertex
        smpl_pts = t_vertices
        _, vert_ids, _ = knn_points(query_pts.unsqueeze(0).float(), smpl_pts.unsqueeze(0).float(), K=1)
        bweights = self.SMPL_NEUTRAL['weights'][vert_ids.squeeze(0)].view(-1,24).cuda()

        # add weights_correction, normalize weights
        # bweights = F.softmax(bweights + 0.2*weights_correction, dim=1)
        bweights = bweights + 0.2 * weights_correction
        bweights = bweights / torch.sum(bweights, dim=-1, keepdim=True)

        ### From Big To T Pose
        big_pose_params = self.big_pose_params(params)
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, big_pose_params)
        A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        query_pts = query_pts - A[:, :3, 3]
        R_inv = torch.inverse(A[:, :3, :3].float())
        query_pts = torch.sum(R_inv * query_pts[:, None], dim=2)

        if self.mean_shape:
            # From mean shape to normal shape
            shapedirs = self.SMPL_NEUTRAL['shapedirs'].cuda()
            shape_offset = (shapedirs * torch.reshape(params['shapes'].cuda(), (10,))).sum(dim=-1)
            query_pts = query_pts + shape_offset[vert_ids.squeeze(0).reshape(-1)]

            posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
            pose_ = params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = 1
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1]).cuda()
            pose_offsets = torch.matmul(pose_feature, posedirs.view(6890*3, -1).transpose(1,0)).view(-1, 3)
            query_pts = query_pts + pose_offsets[vert_ids.squeeze(0).reshape(-1)]

        # get tar_pts, smpl space source pose
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params)
        self.s_A = A
        A = torch.mm(bweights, self.s_A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = torch.sum(A[:, :3, :3] * query_pts[:, None], dim=2)
        smpl_src_pts = can_pts + A[:, :3, 3]
        
        # transform points from the smpl space to the world space
        R_inv = torch.inverse(R)
        world_src_pts = torch.mm(smpl_src_pts, R_inv) + Th
        
        return smpl_src_pts, world_src_pts, bweights

    def forward_fusion(self, query_pts, params, embedding, point_3d_feature, agg_feature=None):
        """(x, x-joint_t, target_f_3d) → deformation Net 1 → X_c""" #39+72+128+32
        query_pts_code = self.pos_enc(query_pts)
        poses = torch.repeat_interleave(params['poses'].reshape(1,72), query_pts_code.shape[0], 0)
        embedding = torch.repeat_interleave(embedding.unsqueeze(0), query_pts_code.shape[0], 0)
        if agg_feature != None:
            f = torch.cat([query_pts_code, poses, embedding, point_3d_feature, agg_feature], dim=-1) # 39 + 72 + 32 + 128 = 271
        else:
            f = torch.cat([query_pts_code, poses, embedding, point_3d_feature], dim=-1) # torch.Size([16384, 669])
        return f
    
    def backward_fusion(self, query_pts, params, embedding, point_3d_feature):
        """(X_c, Pose_s, shape, x-joint_c, f_agg) → deformation Net 2 → X_s"""
        query_pts_code = self.pos_enc(query_pts) # torch.Size([16384, 3])
        poses = torch.repeat_interleave(params['poses'].reshape(1,72), query_pts_code.shape[0], 0)
        embedding = torch.repeat_interleave(embedding.unsqueeze(0), query_pts_code.shape[0], dim=0) # torch.Size([16384, 24, 3])
        f = torch.cat([query_pts_code, poses, embedding, point_3d_feature], dim=-1) # torch.Size([16384, 669])
        return f
    
    def prepare_spconv(self, tp_input, t_vertices=False):
        pre = 't_' if t_vertices else ''
        xyz_input_feature = tp_input[pre+'feature'] #torch.from_numpy(tp_input['feature']).cuda().float()
        voxel_integer_coord = tp_input[pre+'coord'].int() #torch.from_numpy(tp_input['coord']).cuda().float().int()
        coord = torch.zeros_like(voxel_integer_coord[:,0:1])
        voxel_integer_coord = torch.cat((coord, voxel_integer_coord), dim=1)
        out_sh = tp_input[pre+'out_sh'].cpu().numpy().astype(np.int32)
        batch_size = 1
        sparse_smpl_vertices = spconv.SparseConvTensor(xyz_input_feature, voxel_integer_coord, out_sh, batch_size)

        return sparse_smpl_vertices

    def set_extract_mesh(self, flag):
        self.extract_mesh = flag

    def forward(self, sp_input, tp_input, world_query_pts, viewdir):

        ## translate query points from world space target pose to smpl space target pose
        world_query_pts = world_query_pts.squeeze(0)
        viewdir = viewdir.squeeze(0)
        sp_input, tp_input = sequeeze_0(sp_input, tp_input)
        R = tp_input['params']['R']
        Th = tp_input['params']['Th'] #.astype(np.float32)
        smpl_query_pts = torch.mm(world_query_pts - Th, R)
        
        ## encode images
        img_all = sp_input['img_all']
        self.encode_images(img_all)
        
        ## human region sample
        if self.human_sample:
            tar_smpl_pts = tp_input['vertices']
            tar_smpl_pts = torch.mm(tar_smpl_pts - Th, R)
            distance, _, _ = knn_points(smpl_query_pts.unsqueeze(0).float(), tar_smpl_pts.unsqueeze(0).float(), K=1)
            distance = distance.squeeze(0).view(-1)
            pts_mask = torch.zeros_like(smpl_query_pts[:,0]).cuda().int()
            threshold = 0.05 ** 2
            pts_mask[distance < threshold] = 1
            
            zero_pts = torch.zeros(smpl_query_pts.shape).cuda().float()
            smpl_query_pts = smpl_query_pts[pts_mask==1]
            viewdir = viewdir[pts_mask==1]
        else:
            zero_pts = torch.zeros(smpl_query_pts.shape).cuda().float()
            pts_mask = torch.ones_like(smpl_query_pts[:,0]).cuda().int()
        
        ## More Accurate canonical_pts
        if self.correction_field:
            sparse_smpl_vertices = self.prepare_spconv(tp_input, t_vertices=False)
            normalized_source_pts = self.normalize_pts(smpl_query_pts, tp_input['bounds'])
            point_3d_feature_0 = self.encoder_3d(sparse_smpl_vertices, normalized_source_pts.detach())
            
            instance_idx = tp_input['instance_idx'].long()
            embedding = self.latent_codes(instance_idx)
            fused_feature = self.forward_fusion(smpl_query_pts, tp_input['params'], embedding, point_3d_feature_0)
            
            correction = self.forward_deform(fused_feature.float())
            # canonical_pts = self.coarse_deform_target2c(tp_input['params'], tp_input['vertices'], smpl_query_pts + correction)
            canonical_pts = self.coarse_deform_target2c(tp_input['params'], tp_input['vertices'], smpl_query_pts)
            canonical_pts = canonical_pts + correction
        else:
            canonical_pts = self.coarse_deform_target2c(tp_input['params'], tp_input['vertices'], smpl_query_pts).requires_grad_()

        ## Extract Mesh
        if self.extract_mesh or self.mesh_animation:
            canonical_pts = world_query_pts
            pts_mask = torch.ones_like(canonical_pts[:,0]).cuda().int()

        smpl_src_pts, world_src_pts, bweights = self.coarse_deform_c2source(sp_input['params'], sp_input['t_vertices'], canonical_pts, weights_correction=0.)
        
        if self.correction_field:
            sparse_smpl_vertices = self.prepare_spconv(sp_input, t_vertices=False)
            normalized_source_pts = self.normalize_pts(smpl_src_pts, sp_input['bounds'])
            point_3d_feature_0 = self.encoder_3d(sparse_smpl_vertices, normalized_source_pts.detach())
            
            instance_idx = sp_input['instance_idx'].long()
            embedding = self.latent_codes(instance_idx)
            fused_feature = self.backward_fusion(smpl_src_pts.clone(), sp_input['params'], embedding, point_3d_feature_0)
            
            correction_ = self.backward_deform(fused_feature.float())
            smpl_src_pts = smpl_src_pts + correction_
            A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, sp_input['params'])
            R_inv = torch.inverse(R)
            world_src_pts = torch.mm(smpl_src_pts, R_inv) + Th

        if self.smooth_loss and self.training:
            _, vert_ids, _ = knn_points(canonical_pts.unsqueeze(0).float(), sp_input['t_vertices'].unsqueeze(0).float(), K=1)
            if self.smpl_pts_normal==None:
                self.smpl_pts_normal = compute_normal(sp_input['t_vertices'], self.faces)
            nearest_smpl_normal = self.smpl_pts_normal[vert_ids.squeeze(0)].reshape(-1, 3)

        ## Get multi-view pixel-aligned feature four view
        if self.correction_field or tp_input['pose_index']!=sp_input['pose_index']:
            uv = self.projection(world_src_pts, sp_input['R_all'], sp_input['T_all'], sp_input['K_all'])
        else:
            uv = self.projection(world_query_pts[pts_mask==1], sp_input['R_all'], sp_input['T_all'], sp_input['K_all'])
        point_2d_feature_0 = self.encoder_2d.index(uv, self.image_shape)
        point_2d_feature_0 = point_2d_feature_0.transpose(1,2)
        
        if self.append_rgb:
            img = sp_input['img_all']
            uv_ = 2.0 * uv.unsqueeze(2).type(torch.float32)  / self.image_shape.clone().detach().to(uv.device) - 1.0
            samples = grid_sample(img, uv_)
            samples = samples.squeeze(-1).transpose(1,2)
            sh = samples.shape
            samples = self.view_enc(samples.reshape(-1,3)).reshape(sh[0],sh[1],27)
            point_2d_feature_0 = torch.cat((point_2d_feature_0, samples), dim=-1)

        ## Transformer Fusion of multiview Feature
        if self.use_trans and point_2d_feature_0.shape[1]!=0:
            point_2d_feature = point_2d_feature_0.transpose(0,1)
            point_2d_feature = self.transformer(point_2d_feature)
            point_2d_feature_1 = point_2d_feature[:,0,:]
            point_2d_feature_2 = point_2d_feature[:,1,:]
        else:
            point_2d_feature = torch.mean(point_2d_feature_0, dim=0)
            point_2d_feature_1 = point_2d_feature
            point_2d_feature_2 = point_2d_feature

        # Run in Nerf
        x = self.pos_enc(canonical_pts)
        x = torch.cat((x, point_2d_feature_1), dim=1)
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
            h = torch.cat([feature, viewdir, point_2d_feature_2], -1)
        else:
            h = torch.cat([feature, point_2d_feature_2], -1)

        h = self.views_linear(h)
        h = F.relu(h)
        # rgb = self.rgb_linear(h)
        rgb_weights = F.softmax(self.rgb_linear(h),dim=-1).unsqueeze(1)
        img = sp_input['img_all']
        uv_ = 2.0 * uv.unsqueeze(2).type(torch.float32)  / self.image_shape.clone().detach().to(uv.device) - 1.0
        samples = grid_sample(img, uv_)
        samples = samples.squeeze(-1).transpose(1,2).transpose(0,1)
        rgb = torch.bmm(rgb_weights, samples).squeeze(1)
        
        if self.mesh_animation:
            raw = torch.cat([world_src_pts, bweights, rgb, alpha], dim=-1)
            return raw
        if self.extract_mesh:
            raw = torch.cat([rgb, alpha], -1).unsqueeze(0)
            return raw

        if self.human_sample:
            ret_smpl_query_pts, ret_smpl_src_pts = torch.zeros(zero_pts.shape).cuda().float(), torch.zeros(zero_pts.shape).cuda().float()
            ret_smpl_query_pts[pts_mask==1], ret_smpl_src_pts[pts_mask==1] = smpl_query_pts, smpl_src_pts
            ret_correction, ret_correction_ = torch.zeros(zero_pts.shape).cuda().float(), torch.zeros(zero_pts.shape).cuda().float()
            all_occ_normal, all_nearest_smpl_normal = torch.zeros(zero_pts.shape).cuda().float(), torch.zeros(zero_pts.shape).cuda().float()
            if self.correction_field:
                ret_correction[pts_mask==1] = correction
                ret_correction_[pts_mask==1] = correction_

            raw = torch.zeros(zero_pts.shape[0],4).cuda().float()
            raw[pts_mask==1] = torch.cat([rgb, alpha], -1)
            raw[pts_mask==0] = -80 # softplus and sigmoid is outside
            raw = torch.cat([raw, torch.unsqueeze(pts_mask, -1), ret_correction, ret_correction_, ret_smpl_query_pts, ret_smpl_src_pts], -1)

            if self.smooth_loss and self.training:
                intv_flag = (int(sp_input['global_step'].item()) % int(sp_input['smooth_interval'].item()))==0
                if intv_flag:
                    # TODO wide_sigmoid(alpha)
                    # occ_normal = torch.autograd.grad(alpha, [canonical_pts], grad_outputs=torch.ones_like(alpha), create_graph=True)[0]
                    occ_normal = torch.autograd.grad(wide_sigmoid(alpha), [canonical_pts], grad_outputs=torch.ones_like(alpha), create_graph=True)[0]
                    occ_normal = occ_normal / (torch.norm(occ_normal, dim=-1, keepdim=True) + 1e-8)
                    all_occ_normal[pts_mask==1] = occ_normal
                    all_nearest_smpl_normal[pts_mask==1] = nearest_smpl_normal
                    raw = torch.cat([raw, all_occ_normal, all_nearest_smpl_normal], -1)

                    del x, h, point_2d_feature, point_2d_feature_0
                    del feature, point_3d_feature_0, canonical_pts
                    del pts_mask, sp_input, tp_input, world_query_pts, viewdir, occ_normal
        else:
            pts_mask = torch.ones_like(smpl_query_pts[:,0]).cuda().int()
            pts_mask = torch.unsqueeze(pts_mask, -1)
            raw = torch.cat([rgb, alpha, pts_mask, correction], -1)
        
        return raw.unsqueeze(0)

