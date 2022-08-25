import os
import torch
import time
import math
import numpy as np
import pytorch3d
import imageio
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.ops import interpolate_face_attributes
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from pytorch3d.ops.knn import knn_points
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    RasterizationSettings, 
    MeshRasterizer,  
)

import numpy as np
from parser_config import *
from lib.THuman_dataset import THumanDatasetBatch
from torch.utils.data import DataLoader
from lib.h36m_dataset import H36MDatasetBatch
from run_nerf_batch import create_nerf
from lib.run_nerf_helpers import wide_sigmoid, get_transform_params_torch
from lib.run_nerf_helpers import *
from pytorch3d.ops.knn import knn_points
from pandas import read_pickle
import copy

# sys.path.append(os.path.abspath(''))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
os.environ["CUDA_VISIBLE_DEVICES"]="0"

## Load args
parser = config_parser()
global_args = parser.parse_args()

## Create nerf model and test datatset
_, render_kwargs_test, _, _, _ = create_nerf(global_args)
net_fn = render_kwargs_test['network_fn']
net_fn.module.eval()

if global_args.data_set_type in ["H36M_B", "H36M", "H36M_P"]:
    test_set = H36MDatasetBatch(global_args.data_root, split=global_args.test_split, view_num=global_args.view_num,\
        start=240, interval=20, poses_num=20)
        # start=500, interval=100, poses_num=1)
    image_size = 1000

if global_args.data_set_type in ["NeuBody_B"]:
    test_set = H36MDatasetBatch(global_args.data_root, split=global_args.test_split, view_num=global_args.view_num,\
        start=240, interval=20, poses_num=20)
        # start=500, interval=100, poses_num=1)
    image_size = 1000


if global_args.data_set_type in ["THuman_B", "THuman"]:
    test_set = THumanDatasetBatch(global_args.data_root, split=global_args.test_split, view_num=global_args.view_num,\
        start=13, interval=1, poses_num=10)
    image_size = 512

test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
# SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(os.path.join('model/models', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')))

def get_big_pose_params(params):
    big_pose_params = copy.deepcopy(params)
    big_pose_params['poses'] = torch.zeros((1,72))
    big_pose_params['poses'][0, 5] = 45/180*torch.tensor(np.pi)
    big_pose_params['poses'][0, 8] = -45/180*torch.tensor(np.pi)
    big_pose_params['poses'][0, 23] = -30/180*torch.tensor(np.pi)
    big_pose_params['poses'][0, 26] = 30/180*torch.tensor(np.pi)

    return big_pose_params

def coarse_deform_target2c(params, vertices, query_pts, weights, SMPL_NEUTRAL):

        # joints transformation
        A, R, Th, joints = get_transform_params_torch(SMPL_NEUTRAL, params)        
        
        # transform smpl vertices from world space to smpl space
        smpl_pts = torch.mm((vertices - Th), R)
        query_pts = torch.mm((query_pts - Th), R)
        _, vert_ids, _ = knn_points(query_pts.unsqueeze(0).float(), smpl_pts.unsqueeze(0).float(), K=1)
        bweights = weights

        # From smpl space target pose to smpl space canonical pose
        A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = query_pts - A[:, :3, 3]
        R_inv = torch.inverse(A[:, :3, :3].float())
        can_pts = torch.sum(R_inv * can_pts[:, None], dim=2)
        
        if global_args.mean_shape:
            # To mean shape
            shapedirs = SMPL_NEUTRAL['shapedirs'].cuda()
            shape_offset = (shapedirs * torch.reshape(params['shapes'].cuda(), (10,))).sum(dim=-1)
            can_pts = can_pts - shape_offset[vert_ids.squeeze(0).reshape(-1)]

        # From T To Big Pose        
        big_pose_params = get_big_pose_params(params)
        A, R, Th, joints = get_transform_params_torch(SMPL_NEUTRAL, big_pose_params)
        A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = torch.sum(A[:, :3, :3] * can_pts[:, None], dim=2)
        can_pts = can_pts + A[:, :3, 3]

        return can_pts  

def coarse_deform_c2source(params, t_vertices, query_pts, weights, SMPL_NEUTRAL):
        # Find nearest smpl vertex
        smpl_pts = t_vertices
        _, vert_ids, _ = knn_points(query_pts.unsqueeze(0).float(), smpl_pts.unsqueeze(0).float(), K=1)

        # add weights_correction, normalize weights
        bweights = weights

        ### From Big To T Pose
        big_pose_params = get_big_pose_params(params)
        A, R, Th, joints = get_transform_params_torch(SMPL_NEUTRAL, big_pose_params)
        A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        query_pts = query_pts - A[:, :3, 3]
        R_inv = torch.inverse(A[:, :3, :3].float())
        query_pts = torch.sum(R_inv * query_pts[:, None], dim=2)

        if global_args.mean_shape:
            # From mean shape to normal shape
            shapedirs = SMPL_NEUTRAL['shapedirs'].cuda()
            shape_offset = (shapedirs * torch.reshape(params['shapes'].cuda(), (10,))).sum(dim=-1)
            query_pts = query_pts + shape_offset[vert_ids.squeeze(0).reshape(-1)]

        # get tar_pts, smpl space source pose
        A, R, Th, joints = get_transform_params_torch(SMPL_NEUTRAL, params)
        A = torch.mm(bweights, A.reshape(24, -1))
        A = torch.reshape(A, (-1, 4, 4))
        can_pts = torch.sum(A[:, :3, :3] * query_pts[:, None], dim=2)
        smpl_src_pts = can_pts + A[:, :3, 3]
        
        # transform points from the smpl space to the world space
        R_inv = torch.inverse(R)
        world_src_pts = torch.mm(smpl_src_pts, R_inv) + Th
        
        return smpl_src_pts, world_src_pts, bweights

## Complicated Version
# can_verts ->(query)-> skinning weights ->(Expand, Deform)-> tar_verts
# tar_verts ->(Calc intersection)-> tar_inters
# tar_inters ->(Deform back by interpolate skinning weights)-> can_inters
# can_inters ->(Shrink,Run in Nerf) -> rgb

## Single Version
# can_verts ->(query)-> skinning weights ->(Expand, Deform)-> tar_verts, rgb
# tar_verts ->(Calc intersection)-> tar_inters
# interpolate color

# filename = './objs/save/m_X_template.obj'
# filename = './objs/S1_smpl_smooth_normal_align_ft_120.obj' #14407
# filename = './objs/hyd_smpl_smooth_normal_align_ft_120.obj' #14407
filename = './objs/hyd_no_ft_smpl_smooth_normal_align.obj' #14407
filename = "./objs/hyd_append_rgb_N_128.obj"
filename = "./objs/hyd_append_rgb_N_128_ft_120.obj"
view_index = 4 #1 # 4
M, interval = 11, 0.01 #0.005完全不行 0.05还不错，除了有黑边 0.02 看起来不错 THuman_1 5,0.03 / H36M 5, 0.02
intv_list = [[x-M//2 for x in range(M)]]
# intv_list = [[-2, -1, 0., 1, 2]]

verts, triangles, aux = load_obj(filename)
query_pts = verts
sh = query_pts.shape
flat = query_pts.reshape([-1,3]).cuda().float()

net_fn.module.mesh_animation = True
for i, data in enumerate(test_loader):
    data = to_cuda(device, data)
    if i==0:
        sp_input = data
    tp_input = data

    with torch.no_grad():
        chunk = (200000)
        # get mesh vertices skinning weights, rgb, occupancy from source space
        if i==0:
            raw = torch.cat([net_fn(sp_input, sp_input, flat[i:i+chunk], torch.zeros_like(flat[i:i+chunk])) for i in range(0, flat.shape[0], chunk)], 0)
            bweights, rgb_0, alpha_0 = raw[:, 3:27], wide_sigmoid(raw[:, 27:30]), wide_sigmoid(raw[:, 30:31])

        # deform canonical mesh to target space by skinning weights
        # raw = torch.cat([net_fn(tp_input, tp_input, flat[i:i+chunk], torch.zeros_like(flat[i:i+chunk])) for i in range(0, flat.shape[0], chunk)], 0)
        # world_src_pts_ = raw[:, 0:3]
    tp_input_new = sequeeze_0(copy.deepcopy(tp_input))
    _, world_src_pts, _ = coarse_deform_c2source(tp_input_new['params'], tp_input_new['t_vertices'], flat, bweights, net_fn.module.SMPL_NEUTRAL)

    target_pose_obj_filename = filename[0:-4] + "_world_mesh.obj"
    pytorch3d.io.save_obj(target_pose_obj_filename, world_src_pts.cpu(), triangles[0])
    world_mesh = load_objs_as_meshes([target_pose_obj_filename], device=device)

    # # target world smpl mesh
    # SMPL_NEUTRAL = read_pickle(os.path.join('msra_h36m', 'smplx/smpl/SMPL_NEUTRAL.pkl'))
    # smpl_faces = SMPL_NEUTRAL['f']
    # pytorch3d.io.save_obj(filename[0:-4] + "_world_mesh_SMPL.obj", tp_input['vertices'][0].cpu(), torch.from_numpy(smpl_faces.astype(np.int32)))

    R = tp_input["R_all"][:, view_index, ...].clone()
    R[0,0] = -R[0,0]
    R[0,1] = -R[0,1]
    R[0,2] = R[0,2]
    R = R.transpose(1,2) # camera coordinate axis are not aligned

    T = tp_input["T_all"][:, view_index, ...].reshape(-1,3).clone()
    T[0, 0] = -T[0, 0]
    T[0, 1] = -T[0, 1]
    T[0, 2] = T[0, 2] #- 1.0 # 注意这种使用方式，会更改sp_input["R_all"][:, 2, ...]， sp_input["T_all"][:, 2, ...].reshape(-1,3)

    K = tp_input["K_all"][:, view_index, ...][0].clone()
    fov = 2.*math.atan((image_size//2) / K[0][0])
    
    fov=2.*math.atan(0.5*(0.8/1.0))
    """
        ## H36M: 正面
        # R = torch.tensor([[
        #     [-1,0,0],
        #     [0,0,1],
        #     [0,1,0]
        # ]]) # Extrinsic: （行坐标）描述的是 相机坐标系xyz轴 用 世界坐标系xyz轴 表示 
        # R = R.transpose(1,2) # Camera Pose
        # T = torch.tensor([[-0.5,-1.0,2.5]]) # Extricsic: 世界坐标系origin在相机坐标系中的位置

        ## H36M: 侧面
        # R = torch.tensor([[
        #     [0,1,0],
        #     [0,0,1],
        #     [1,0,0]
        # ]]) # Extrinsic: （行坐标）描述的是 相机坐标系xyz轴 用 世界坐标系xyz轴 表示 
        # R = R.transpose(1,2) # Camera Pose
        # T = torch.tensor([[0,-1.0,2.0]]) # Extricsic: 世界坐标系origin在相机坐标系中的位置
    """
    ## Pytorch3D 的 K (N,4,4)，并不是内参矩阵， 和ndc坐标系有关系，需要后面学习一下
    cameras = FoVPerspectiveCameras(znear=0.01,zfar=50, fov=fov, degrees=False, device=device, R=R, T=T)#, fov=2. * math.atan(0.5*(0.8/1.0)), znear=0.01, zfar=100)
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragement = rasterizer(world_mesh)
    pix_to_face, _, bary_coords, _ = fragement[0], fragement[1], fragement[2], fragement[3]

    ## Calculate intersection_coords by functions used Pytorch3D Shader
    verts = world_mesh.verts_packed()
    faces = world_mesh.faces_packed()

    ## Directly interpolate rgb
    faces_verts_rgb = rgb_0[faces]
    intersection_rgb = interpolate_face_attributes(pix_to_face, bary_coords, faces_verts_rgb) 
    intersection_rgb = intersection_rgb.reshape(image_size,image_size,3)
    rgb8 = to8b(intersection_rgb.cpu().numpy())

    target_s = tp_input['img_all'][0, view_index]
    gt_rgb = target_s.permute(1,2,0)
    gt_rgb8 = to8b(gt_rgb.cpu().numpy())
    rgb8 = np.concatenate([rgb8, gt_rgb8], axis=1)

    print(target_pose_obj_filename[:-4]+('/_%d.'%i)+'png')
    os.makedirs(target_pose_obj_filename[:-4], exist_ok=True)
    imageio.imwrite(target_pose_obj_filename[:-4]+('/_%d.'%i)+'png', rgb8)

    # if the pixel has no intersection, pixel_coords values are set to -1
    faces_verts_coords = verts[faces]
    intersection_coords = interpolate_face_attributes(pix_to_face, bary_coords, faces_verts_coords) 
    intersection_coords = intersection_coords.reshape(image_size,image_size,3)

    # Template: find ray dir in sp input
    origin_R = tp_input["R_all"][:, view_index, ...][0]
    origin_T = tp_input["T_all"][:, view_index, ...][0]
    origin = -origin_R@origin_T
    mask = (pix_to_face[0]!=-1.).squeeze(-1) # torch.Size([512, 512])
    ray_dir = torch.zeros_like(intersection_coords)
    ray_dir[mask] = intersection_coords[mask] -  origin.reshape(3,).float()
    ray_dir[mask] = ray_dir[mask] / torch.norm(ray_dir[mask], dim=-1, keepdim=True)
    
    # trace 5 points along the ray around the intersection
    
    # intv_list = [[-1, 0., 1,]]
    delta = ray_dir.unsqueeze(0).expand(M,-1,-1,-1) * torch.tensor(intv_list).reshape(M,1,1,1) * interval
    N_samples = intersection_coords.unsqueeze(0).expand(M,-1,-1,-1) + delta # torch.Size([5, 512, 512, 3])

    # backward LBS 5 points
    faces_verts_weights = bweights[faces]
    intersection_weights = interpolate_face_attributes(pix_to_face, bary_coords, faces_verts_weights)
    intersection_weights = intersection_weights.reshape(image_size,image_size,24)

    # SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(os.path.join('model/models', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')))
    valid_pts = N_samples[mask.unsqueeze(0).expand(M,-1,-1)]
    valid_weights = intersection_weights.unsqueeze(0).expand(M,-1,-1,-1)[mask.unsqueeze(0).expand(M,-1,-1)]
    tp_input_new = sequeeze_0(copy.deepcopy(tp_input))
    can_valid_pts = coarse_deform_target2c(tp_input_new['params'], tp_input_new['vertices'], valid_pts, valid_weights, net_fn.module.SMPL_NEUTRAL)
    with torch.no_grad():
        chunk = (200000)
        raw = torch.cat([net_fn(sp_input, sp_input, can_valid_pts[i:i+chunk], torch.zeros_like(can_valid_pts[i:i+chunk])) for i in range(0, can_valid_pts.shape[0], chunk)], 0)
        rgb, alpha = wide_sigmoid(raw[:, 27:30]), wide_sigmoid(raw[:, 30:31])

    # rgb.shape torch.Size([1, 500, 64, 3]) alpha.shape torch.Size([1, 500, 64])
    alpha = alpha.reshape(1, M, -1).transpose(1,2)
    rgb = rgb.reshape(1, M, -1, 3).transpose(1,2) # 按照维度顺序 reshape
    ##
    # rgb[0,:,2,:] = intersection_rgb[mask]
    # alpha[0,:,2] = 0.5
    ##
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], alpha.shape[1], 1)), 1.-alpha + 1e-10], -1), -1)[:, :, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)

    rgb_img = torch.zeros_like(N_samples[0])
    rgb_img[mask] = rgb_map.squeeze(0)

    ## Save image
    rgb8 = to8b(rgb_img.cpu().numpy())
    print(target_pose_obj_filename[:-4]+('/%d.'%i)+'png')
    os.makedirs(target_pose_obj_filename[:-4], exist_ok=True)
    imageio.imwrite(target_pose_obj_filename[:-4]+('/%d.'%i)+'png', rgb8)
    



# index = uv[0].cpu().numpy().astype(np.int)
# msk = np.zeros((1000, 1000))
# msk[index[:,1], index[:,0]] = 255
# imageio.imwrite('objs/msk.png', msk)

# for i in range(uv[0].cpu().numpy().shape[0]):
#     index = uv[0].cpu().numpy()[i].astype(np.int32)
#     msk[index[1], index[0]] = 255
# imageio.imwrite('objs/msk.png', msk)

# img = sp_input['img_all']
# uv_ = 2.0 * uv.unsqueeze(2).type(torch.float32)  / self.image_shape.clone().detach().to(uv.device) - 1.0
# samples = grid_sample(img, uv_)
# msk = np.zeros((1000, 1000, 3))
# index = uv[0].detach().cpu().numpy().astype(np.int)
# msk[index[:,1], index[:,0]] = samples[0].squeeze(-1).transpose(0,1).detach().cpu().numpy()
# imageio.imwrite('objs/msk_rgb.png', msk)

# mask = np.ones((512, 512))*100
# mask[mask_at_box[j].cpu().numpy()] = 200
# mask[msk[j].cpu().numpy().astype(np.int32)==1] = 255
# psnr = mse2psnr(img2mse(img_pred[msk[j]==1], gt_img[msk[j]==1]))

