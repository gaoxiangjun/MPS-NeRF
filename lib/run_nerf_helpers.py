import enum
from numpy.lib.npyio import save
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd.profiler as profiler
import random
import cv2
from memory_profiler import profile
import os
from PIL import Image, ImageDraw, ImageFont
# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
shifted_softplus = lambda x : F.softplus(x-1)
wide_sigmoid = lambda x : ((1 + 2*0.0001)*torch.sigmoid(x) - 0.0001)


def extended_img2mse(x,y,pose_index_1, pose_index_2):
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    flag = (pose_index_1 == pose_index_2)
    if flag.sum()==0:
        mse = 0.0
    else:
        mse = torch.mean((x[flag] - y[flag]) ** 2)
    return mse

def images_to_video(image_folder, video_name=None, images=None, fps=1):
    video_name = os.path.join(image_folder, video_name+".wmv")
    frame = cv2.imread(os.path.join(image_folder, images[0]))# [150:662, 250:762]
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, fps, (width, height))
    for image in images:
        video.write(cv2.resize(cv2.imread(os.path.join(image_folder, image)), (width, height)))
    # cv2.destroyAllWindows()
    video.release()

# import imageio
# from PIL import Image
# import os 
# import numpy as np
# import cv2

# novel_pose = "_01.png"
# savedir = "/home/v-xgao/local/human_nerf/logs/H36M_sample_new_exclude_S11_all_poses/testset_002400_more"
# images = [img for img in os.listdir(savedir) if img.endswith(novel_pose)]
# images.sort()
# video_name = "new"
# # images_to_video(savedir, video_name="novel_pose", images=images, fps=1)

# video_name = os.path.join(savedir, video_name+".wmv")
# writer = imageio.get_writer(video_name, fps=1)
# for image in images:
#     pil_img = np.array(Image.open(os.path.join(savedir, image)))
#     im_cv = cv2.imread(os.path.join(savedir, image))
#     im_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
#     writer.append_data(pil_img)
# writer.close()

    
def image_add_text(img_path, text, left, top, text_color=(255, 0, 0), text_size=13):
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype("DejaVuSerif.ttf", text_size, encoding="utf-8")
    draw.text((left, top), text, text_color, font=fontStyle)
    return img

# random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_numpy(sp_input, tp_input=None):
    for key in sp_input.keys():
        if torch.is_tensor(sp_input[key]):
            # sp_input[key] = sp_input[key].repeat(repeat_num)
            sp_input[key] = sp_input[key].cpu().numpy()
            sp_input[key] = np.squeeze(sp_input[key], 0)
        if isinstance(sp_input[key], list):
            for i, data in enumerate(sp_input[key]):
                sp_input[key][i] = data.cpu().numpy()
                sp_input[key][i] = np.squeeze(sp_input[key][i], 0)
        if key=='params':
            for key1 in sp_input['params']:
                if torch.is_tensor(sp_input['params'][key1]):
                    sp_input['params'][key1] = sp_input['params'][key1].cpu().numpy() 
                    sp_input['params'][key1] = np.squeeze(sp_input['params'][key1], 0)
    
    if tp_input==None:
        return sp_input
    
    for key in tp_input.keys():
        if torch.is_tensor(tp_input[key]):
            tp_input[key] = tp_input[key].cpu().numpy()
            tp_input[key] = np.squeeze(tp_input[key], 0)
        if isinstance(tp_input[key], list):
            for i, data in enumerate(tp_input[key]):
                tp_input[key][i] = data.cpu().numpy()
                tp_input[key][i] = np.squeeze(tp_input[key][i], 0)
        if key=='params':
            for key1 in tp_input['params']:
                if torch.is_tensor(tp_input['params'][key1]):
                    tp_input['params'][key1] = tp_input['params'][key1].cpu().numpy() 
                    tp_input['params'][key1] = np.squeeze(tp_input['params'][key1], 0)

    return sp_input, tp_input


def to_cuda(device, sp_input, tp_input=None):
    for key in sp_input.keys():
        if torch.is_tensor(sp_input[key]):
            sp_input[key] = sp_input[key].to(device)
        if key=='params':
            for key1 in sp_input['params']:
                if torch.is_tensor(sp_input['params'][key1]):
                    sp_input['params'][key1] = sp_input['params'][key1].to(device)
    
    if tp_input==None:
        return sp_input
    
    for key in tp_input.keys():
        if torch.is_tensor(tp_input[key]):
            tp_input[key] = tp_input[key].to(device) 
        if key=='params':
            for key1 in tp_input['params']:
                if torch.is_tensor(tp_input['params'][key1]):
                    tp_input['params'][key1] = tp_input['params'][key1].to(device)

    return sp_input, tp_input

def SMPL_to_tensor(params):
    key_ = ['v_template', 'shapedirs', 'J_regressor', 'kintree_table', 'f', 'weights', "posedirs"]
    for key1 in key_:
        if key1 == 'J_regressor':
            params[key1] = torch.tensor(params[key1].toarray().astype(float)).float()
        elif key1 == 'kintree_table' or key1 == 'f':
            params[key1] = torch.tensor(np.array(params[key1]).astype(float)).long()
        else:
            params[key1] = torch.tensor(np.array(params[key1]).astype(float)).float()
    return params

def sequeeze_0(sp_input, tp_input=None):
    for key in sp_input.keys():
        if torch.is_tensor(sp_input[key]):
            # sp_input[key] = sp_input[key].repeat(repeat_num)
            sp_input[key] = torch.squeeze(sp_input[key], 0).float()
        if key=='params':
            for key1 in sp_input['params']:
                if torch.is_tensor(sp_input['params'][key1]):
                    sp_input['params'][key1] = torch.squeeze(sp_input['params'][key1], 0).float()    
    if tp_input==None:
        return sp_input
    
    for key in tp_input.keys():
        if torch.is_tensor(tp_input[key]):
            tp_input[key] = torch.squeeze(tp_input[key], 0).float()
        if key=='params':
            for key1 in tp_input['params']:
                if torch.is_tensor(tp_input['params'][key1]):
                    tp_input['params'][key1] = torch.squeeze(tp_input['params'][key1], 0).float()

    return sp_input, tp_input
  
def batch_rodrigues_torch(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = torch.norm(poses + 1e-8, p=2, dim=1, keepdim=True)
    rot_dir = poses / angle

    cos = torch.cos(angle)[:, None]
    sin = torch.sin(angle)[:, None]

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros([batch_size, 1])
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1)
    K = K.reshape([batch_size, 3, 3])

    ident = torch.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    return rot_mat


def get_rigid_transformation_torch(rot_mats, joints, parents):
    """
    rot_mats: 24 x 3 x 3
    joints: 24 x 3
    parents: 24
    """
    # obtain the relative joints
    rel_joints = joints.clone()
    rel_joints[1:] -= joints[parents[1:]]

    # create the transformation matrix
    transforms_mat = torch.cat([rot_mats, rel_joints[..., None]], dim=2)
    padding = torch.zeros([24, 1, 4]).cuda()
    padding[..., 3] = 1
    transforms_mat = torch.cat([transforms_mat, padding], dim=1)

    # rotate each part
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=0)

    # obtain the rigid transformation
    padding = torch.zeros([24, 1]).cuda()
    joints_homogen = torch.cat([joints, padding], dim=1)
    rel_joints = torch.sum(transforms * joints_homogen[:, None], dim=2)
    transforms[..., 3] = transforms[..., 3] - rel_joints

    return transforms

# @profile
def get_transform_params_torch(smpl, params):
    """ obtain the transformation parameters for linear blend skinning
    """
    device = params['shapes'].device
    v_template = smpl['v_template'].to(device)

    # add shape blend shapes
    shapedirs = smpl['shapedirs'].to(device)
    betas = params['shapes']
    v_shaped = v_template + torch.sum(shapedirs * betas[None], axis=2).float()

    # add pose blend shapes
    poses = params['poses'].reshape(-1, 3)
    # 24 x 3 x 3
    rot_mats = batch_rodrigues_torch(poses)

    # obtain the joints
    joints = torch.matmul(smpl['J_regressor'].to(device), v_shaped) # v

    # obtain the rigid transformation
    parents = smpl['kintree_table'][0].to(device)
    A = get_rigid_transformation_torch(rot_mats, joints, parents)

    # apply global transformation
    R = params['R'] #.to(device)
    Th = params['Th'] #.to(device)

    return A, R, Th, joints


def remove_first_dim(sp_input, tp_input):
    for key in sp_input.keys():
        if torch.is_tensor(sp_input[key]):
            sp_input[key] = torch.squeeze(sp_input[key], 0)
        if isinstance(sp_input[key], list):
            for i, data in enumerate(sp_input[key]):
                sp_input[key][i] = torch.squeeze(sp_input[key][i], 0)
        if key=='params':
            for key1 in sp_input['params']:
                if torch.is_tensor(sp_input['params'][key1]):
                    sp_input['params'][key1] = torch.squeeze(sp_input['params'][key1], 0)

    for key in tp_input.keys():
        if torch.is_tensor(tp_input[key]):
            tp_input[key] = torch.squeeze(tp_input[key], 0)
        if isinstance(tp_input[key], list):
            for i, data in enumerate(tp_input[key]):
                tp_input[key][i] = torch.squeeze(tp_input[key][i], 0)
        if key=='params':
            for key1 in tp_input['params']:
                if torch.is_tensor(tp_input['params'][key1]):
                    tp_input['params'][key1] = torch.squeeze(tp_input['params'][key1], 0)

    return sp_input, tp_input

def repeat_first_dim(sp_input, tp_input, repeat_num=8):
    for key in sp_input.keys():
        if torch.is_tensor(sp_input[key]):
            sp_input[key] = sp_input[key].repeat(repeat_num)
        if isinstance(sp_input[key], list):
            for i, data in enumerate(sp_input[key]):
                sp_input[key][i] = sp_input[key][i].repeat(repeat_num)
        if key=='params':
            for key1 in sp_input['params']:
                if torch.is_tensor(sp_input['params'][key1]):
                    sp_input['params'][key1].repeat(repeat_num)

    for key in tp_input.keys():
        if torch.is_tensor(tp_input[key]):
            tp_input[key] = tp_input[key].repeat(repeat_num)
        if isinstance(tp_input[key], list):
            for i, data in enumerate(tp_input[key]):
                tp_input[key][i] = tp_input[key][i].repeat(repeat_num)
        if key=='params':
            for key1 in tp_input['params']:
                if torch.is_tensor(tp_input['params'][key1]):
                    tp_input['params'][key1].repeat(repeat_num)

    # import copy
    # sp, tp = copy.deepcopy(sp_input), copy.deepcopy(tp_input)
    # for key in ['params', 'rgb_all', 'ray_o_all', 'ray_d_all', 'near_all', 'far_all', 'mask_at_box_all']:
    #     sp.pop(key)
    #     tp.pop(key)

    return sp_input, tp_input

class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        with profiler.record_function("positional_enc"):
            embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
            embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
            if x.shape[0]==0:
                embed = embed.view(x.shape[0], self.num_freqs*6)
            else:
                embed = embed.view(x.shape[0], -1)
                
            if self.include_input:
                embed = torch.cat((x, embed), dim=-1)
            return embed


class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [DenseLayer(input_ch, W, activation="relu")] + [DenseLayer(W, W, activation="relu") if i not in self.skips else DenseLayer(W + input_ch, W, activation="relu") for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([DenseLayer(input_ch_views + W, W//2, activation="relu")])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = DenseLayer(W, W, activation="linear")
            self.alpha_linear = DenseLayer(W, 1, activation="linear")
            self.rgb_linear = DenseLayer(W//2, 3, activation="linear")
        else:
            self.output_linear = DenseLayer(W, output_ch, activation="linear")

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
