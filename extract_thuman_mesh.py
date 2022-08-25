import os
from time import time
import torch
import numpy as np
from parser_config import *
from lib.THuman_dataset import THumanDatasetBatch
from torch.utils.data import DataLoader
from run_nerf_batch import create_nerf
import mcubes
from lib.run_nerf_helpers import wide_sigmoid
from lib.run_nerf_helpers import *
from pytorch3d.ops.knn import knn_points
from pandas import read_pickle
import time
import imageio
start = time.time()


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

## General setup for GPU device and default tensor type.
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

## Load args
parser = config_parser()
global_args = parser.parse_args()
# global_args.view_num = 3
# global_args.expname = "Final_THuman1_25_person_{}_view_no_mean_fm_all_loss".format(global_args.view_num)
# global_args.expname = "Final_THuman1_25_person_{}_view_no_mean_fm_no_shape_loss".format(global_args.view_num)
# global_args.expname = "Final_THuman1_25_person_{}_view_no_mean_fm_no_smooth_loss".format(global_args.view_num)

## Create nerf model and test datatset
_, render_kwargs_test, _, _, _ = create_nerf(global_args, device=device)
net_fn = render_kwargs_test['network_fn']
net_fn.module.eval()
ratio = 1.0
data_root_list = [
     "./data/THuman/nerf_data_/results_gyx_20181012_sty_1_M",
     "./data/THuman/nerf_data_/results_gyx_20181012_xsx_2_M",
     "./data/THuman/nerf_data_/results_gyx_20181013_hyd_1_M",
     "./data/THuman/nerf_data_/results_gyx_20181012_lw_2_F",
     "./data/THuman/nerf_data_/results_gyx_20181013_xyz_1_F",
]
input_pose_list = [20, 24, 27, 25, 28]
novel_pose_num = 5
threshold = 30 #0.5 # 0.4
N = 256


for p, data_root in enumerate(data_root_list):

    test_set = THumanDatasetBatch(data_root, split=global_args.test_split, view_num=global_args.view_num,\
        start=input_pose_list[p], interval=1, poses_num=novel_pose_num + 1)#hyd 13 22 # lc 5 
    H,W = int(512 * ratio), int(512 * ratio)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    ## Estimate occupancy for grid
    print("Begin to estimate occupancy! ")
    for i, data in enumerate(test_loader):
        # print(i)
        data = to_cuda(device, data)
        if i==0:
            sp_input = data
        tp_input = data

        ## Sample a grid in canonical space
        # H, W, focal, down = 512, 512, 640, 1
        can_flag = False
        ## canonical pose
        if can_flag:
            t_vertices = np.load(os.path.join('m_X_template_tvertices.npy'))
            t_1 = np.linspace(-1.0, 1.0, N) # x
            t_2 = np.linspace(-1.0, 1.0, N)   # y
            t_3 = np.linspace(-0.25, 0.25, N//4) # z
            START = np.array([-1.0, -1.0, -0.25])
            SIZE = np.array([2.0, 2.0, 0.5])
            RANGE = np.array([N, N, N//4])
        else:
            ## target pose
            t_vertices = tp_input['vertices'].squeeze(0).cpu().numpy()
            # t_vertices = np.load(os.path.join('m_X_template_tvertices.npy'))
            t_1 = np.linspace(0.0, 2.0, N)   # y
            t_2 = np.linspace(0.6, 2.6, N) # x
            t_3 = np.linspace(0.0, 2.0, N) # z
            START = np.array([0.0, 0.6, 0.0])
            SIZE = np.array([2.0, 2.0, 2.0])
            RANGE = np.array([N, N, N])

        query_pts = np.stack(np.meshgrid(t_1, t_2, t_3), -1).astype(np.float32)
        sh = query_pts.shape
        flat = torch.from_numpy(query_pts.reshape([-1,3])).cuda().float()

        if can_flag:
            net_fn.module.set_extract_mesh(True)
        with torch.no_grad():
            chunk = (200000 * 5)
            raw = torch.cat([net_fn(sp_input, tp_input, flat[i:i+chunk], torch.zeros_like(flat[i:i+chunk]))[0, ..., 0:4] for i in range(0, flat.shape[0], chunk)], 0)
            raw = torch.reshape(raw, list(sh[:-1]) + [-1])
            # occupancy = wide_sigmoid(raw[...,3])
            occupancy = shifted_softplus(raw[...,3])


        smpl_path = os.path.join('assets', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl') if data_root[-1]=="M" else os.path.join('assets', 'basicmodel_f_lbs_10_207_0_v1.0.0.pkl')
        SMPL_NEUTRAL = read_pickle(smpl_path)
        # t_vertices = tp_input['vertices']
        t_vertices = torch.from_numpy(t_vertices).cuda().float()
        distance, vert_ids, _ = knn_points(flat.unsqueeze(0).float(), t_vertices.unsqueeze(0).float(), K=1)
        distance = distance.squeeze(0).view(-1)
        pts_mask = torch.zeros_like(flat[:,0]).cuda().int()
        dis_threshold = 0.05 ** 2 # 0.03
        pts_mask[distance < dis_threshold] = 1
        pts_mask = torch.reshape(pts_mask, list(sh[:-1]))

        Set_inside_occ_0 = False
        if Set_inside_occ_0:
            ### Set_inside_occ_0
            occupancy[pts_mask==0] = 0.
            occupancy = occupancy.cpu().numpy()
        else:
            ### Set_inside_occ_1
            faces = SMPL_NEUTRAL['f'].astype(np.int32)
            smpl_pts_normal = compute_normal(t_vertices, faces)
            ## set_inside_occ_1_by_mean_normal
            distance, vert_ids, _ = knn_points(flat.unsqueeze(0).float(), t_vertices.unsqueeze(0).float(), K=5)
            pts_dir = flat - t_vertices[vert_ids.squeeze(0)].mean(dim=1)
            pts_dir = pts_dir / torch.norm(pts_dir, dim=-1, keepdim=True)
            ## set_inside_occ_1_by_nearest_normal
            # pts_dir = flat - t_vertices[vert_ids.squeeze(0).cpu().numpy().reshape(-1)]
            # face_normal = smpl_pts_normal[vert_ids.squeeze(0).cpu().numpy().reshape(-1)]
            face_normal = smpl_pts_normal[vert_ids.squeeze(0)].mean(dim=1)
            outside_msk = ((pts_dir * face_normal).sum(dim=-1) > 0)
            outside_msk = torch.reshape(outside_msk, list(sh[:-1]))
            occupancy[pts_mask==0] = 0.
            # occupancy[(pts_mask==0) & (outside_msk==0)] = 1.
            occupancy[(pts_mask==0) & (outside_msk==0)] = 100.
            occupancy = occupancy.cpu().numpy()

        # print('fraction occupied', np.mean(occupancy > threshold))
        vertices, triangles = mcubes.marching_cubes(occupancy, threshold)

        ### For canonical space
        if can_flag:
            vertices_ = START + vertices*(SIZE/RANGE) #[198,127,31]
            vertices_trans = np.zeros_like(vertices_)
            vertices_trans[:,0] = -vertices_[:,1]
            vertices_trans[:,1] = vertices_[:,0]
            vertices_trans[:,2] = vertices_[:,2]
            
            sp_pose = int(sp_input['pose_index'].cpu().numpy()) + input_pose_list[p]
            tp_pose = int(tp_input['pose_index'].cpu().numpy()) + input_pose_list[p]
            name = os.path.basename(data_root)
            os.makedirs('objs/THuman/{}'.format(global_args.expname), exist_ok=True)
            os.makedirs('objs/THuman/{}/{}'.format(global_args.expname, name), exist_ok=True)
            mcubes.export_obj(vertices_trans, triangles, 'objs/THuman/{}/{}/canonical_3_view_input_{:03d}_output_{:03d}.obj'.format(global_args.expname, name, sp_pose, tp_pose))

        else:
            ### For target soace
            vertices_trans = np.zeros_like(vertices)
            vertices_trans[:,0] = START[0] + vertices[:,1]*(SIZE[0]/RANGE[0]) # y
            vertices_trans[:,1] = START[1] + vertices[:,0]*(SIZE[1]/RANGE[1]) # x
            vertices_trans[:,2] = START[2] + vertices[:,2]*(SIZE[2]/RANGE[2]) # z
            new_triangles = np.zeros_like(triangles)
            new_triangles = triangles[:, ::-1] 
            # mcubes.export_obj(vertices_trans, new_triangles, 'objs/8_view_coarse_smooth_22_14750_thre_30.obj')
            
            sp_pose = int(sp_input['pose_index'].cpu().numpy()) + input_pose_list[p]
            tp_pose = int(tp_input['pose_index'].cpu().numpy()) + input_pose_list[p]
            name = os.path.basename(data_root)
            os.makedirs('objs/THuman/{}'.format(global_args.expname), exist_ok=True)
            os.makedirs('objs/THuman/{}/{}'.format(global_args.expname, name), exist_ok=True)
            mcubes.export_obj(vertices_trans, new_triangles, 'objs/THuman/{}/{}/novel_3_view_input_{:03d}_output_{:03d}.obj'.format(global_args.expname, name, sp_pose, tp_pose))

            y,x = 0,130 #- 10
            h,w = 512,256 #+ 80# THuman
            if i==0:
                for j in range(sp_input['img_all'].shape[1]):
                    input_img = sp_input['img_all'][0][j].cpu().numpy().transpose(1,2,0) * 255.
                    filename = os.path.join('objs/THuman/{}/{}/input_{:03d}_view_{:03d}.png'.format(global_args.expname, name, sp_pose, j))
                    crop_img = input_img[y:y+h, x:x+w]
                    imageio.imwrite(filename, to8b(crop_img/255.))
            for j in range(tp_input['img_all'].shape[1]):
                output_img = tp_input['img_all'][0][j].cpu().numpy().transpose(1,2,0) * 255.
                filename = os.path.join('objs/THuman/{}/{}/output_{:03d}_view_{:03d}.png'.format(global_args.expname, name, tp_pose, j))
                crop_img = output_img[y:y+h, x:x+w]
                imageio.imwrite(filename, to8b(crop_img/255.))
            
        print('done', vertices.shape, triangles.shape, "Time: ", time.time() - start)
        start = time.time()
