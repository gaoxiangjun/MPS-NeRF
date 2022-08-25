from operator import pos
from numpy.lib.polynomial import roots
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import imageio
import cv2
import time
import lib.if_nerf_data_utils as if_nerf_dutils
from lib.if_nerf_data_utils import get_rays, get_bound_2d_mask, get_near_far
import lib.render_utils as render_utils


class H36MDataset(Dataset):
    def __init__(self, data_root, split='test', view_num=3, border=5, N_rand=1024*32, multi_person=False, num_instance=1, img_scaling=1.0,
                start=0, interval=10, poses_num=100):
        super(H36MDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        if view_num==4:
            self.input_view =  [0,1,2,3] # [x for x in range(24)] # [0,1,2,3]
        elif view_num==3:
            self.input_view = [0,2,3]
        else:
            self.input_view =  [0,1,2,3]
        self.train_view = [0,1,2,3]
        self.test_view =  [0,1,2,3]
        self.output_view = self.train_view if split == 'train' else self.test_view

        i = start
        i_intv = interval # 5, 10, 30
        ni = poses_num # 300, 150, 50
        self.ims = np.array([
            np.array(ims_data['ims'])[self.view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ])# .ravel() # image_name shape (150, 4)
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[self.view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ])# .ravel()
        self.nrays = N_rand
        self.border = border
        self.image_scaling = img_scaling
        self.root_list = [data_root]

    def get_mask(self, index, view_index):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[index][view_index])[:-4] + '.png'
        msk_cihp = imageio.imread(msk_path)
        msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp

        border = self.border
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 100

        return msk

    def prepare_input_t(self, t_vertices_path):
        vertices_path = t_vertices_path
        xyz = np.load(vertices_path).astype(np.float32)
        vertices = xyz

        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        
        feature = cxyz
        # feature = np.ones((6890,1)) * 0.01

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array([0.005, 0.005, 0.005])
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, bounds

    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, 'new_vertices', '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        vertices = xyz
        # nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate 
        params_path = os.path.join(self.data_root, "new_params", '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        # nxyz = nxyz.astype(np.float32)
        # feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)
        feature = cxyz

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array([0.005, 0.005, 0.005])
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, world_bounds, bounds, Rh, Th, vertices, params # , center, rot, trans

    def __getitem__(self, index):
        img_path_list = self.ims[index]
        all_img_path = []
        for i, path in enumerate(img_path_list):
            all_img_path.append(os.path.join(self.data_root, path))
        
        img_all, K_all, R_all, T_all, rgb_all, ray_o_all, ray_d_all, near_all, far_all, msk_all = [], [], [], [], [], [], [], [], [], []
        mask_at_box_all, bkgd_msk_all, img_ray_d_all = [], [], []
        for view_index in self.view:
            # Load image, mask, K, D, R, T
            img_path = all_img_path[view_index]
            img = imageio.imread(img_path).astype(np.float32) / 255.
            msk = self.get_mask(index, view_index)
            # img = cv2.resize(img, (1000, 1002), interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (1000, 1000), interpolation=cv2.INTER_NEAREST)
            cam_ind = view_index
            K = np.array(self.cams['K'][cam_ind])
            D = np.array(self.cams['D'][cam_ind])
            img = cv2.undistort(img, K, D)
            msk = cv2.undistort(msk, K, D)
            R = np.array(self.cams['R'][cam_ind])
            T = np.array(self.cams['T'][cam_ind]) / 1000.

            # Reduce the image resolution by ratio, then remove the back ground
            ratio = self.image_scaling
            if ratio != 1.:
                H, W = int(img.shape[0] * ratio), int(img.shape[1] * ratio)
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                mask_bkgd = True
                if mask_bkgd:
                    img[msk == 0] = 0
                K[:2] = K[:2] * ratio
            else:
                mask_bkgd = True
                if mask_bkgd:
                    img[msk == 0] = 0
            
            # Prepare the smpl input, including the current pose and canonical pose
            if view_index == 0:
                i = int(os.path.basename(img_path)[:-4])
                feature, coord, out_sh, world_bounds, bounds, Rh, Th, vertices, params = self.prepare_input(i)                
                path = os.path.join(self.data_root, 'n_X_template_tvertices.npy') # before_200_45_big_pose_tvertices # 45_big_pose_tvertices # posing_tvertices
                smpl_R = cv2.Rodrigues(Rh)[0].astype(np.float32)
                params["R"] = cv2.Rodrigues(params['Rh'])[0].astype(np.float32)
                t_vertices = np.load(path)
                t_feature, t_coord, t_out_sh, t_bounds = self.prepare_input_t(t_vertices_path=path)
            
            # Sample rays in target space world coordinate
            rgb, ray_o, ray_d, near, far, coord_, mask_at_box, bkgd_msk,  img_ray_d = if_nerf_dutils.sample_ray_h36m(
                    img, msk, K, R, T, world_bounds, self.nrays, self.split)

            # Pack all inputs of all views
            img = np.transpose(img, (2,0,1))
            img_ray_d = np.transpose(img_ray_d, (2,0,1))
            if view_index<len(self.train_view):
                img_all.append(img)
                img_ray_d_all.append(img_ray_d)
                K_all.append(K)
                R_all.append(R)
                T_all.append(T)
            msk_all.append(msk)
            rgb_all.append(rgb)
            ray_o_all.append(ray_o)
            ray_d_all.append(ray_d)
            near_all.append(near)
            far_all.append(far)
            mask_at_box_all.append(mask_at_box)
            bkgd_msk_all.append(bkgd_msk)

        img_all = np.stack(img_all, axis=0)
        img_ray_d_all = np.stack(img_ray_d_all, axis=0)
        msk_all = np.stack(msk_all, axis=0)
        K_all = np.stack(K_all, axis=0)
        R_all = np.stack(R_all, axis=0)
        T_all = np.stack(T_all, axis=0)
        

        ret = {
            'pose_index': index,
            'cam_ind': cam_ind,
            't_vertices': t_vertices,
            "instance_idx": 0,

            'feature': feature, # smpl pose space xyz
            'coord': coord,    # smpl pose space coords
            'bounds': bounds, # smpl pose space bounds
            'out_sh': out_sh,   # smpl pose space out_sh

            't_feature': t_feature, # smpl pose space xyz
            't_coord': t_coord,    # smpl pose space coords
            't_bounds': t_bounds, # smpl pose space bounds
            't_out_sh': t_out_sh,   # smpl pose space out_sh
            
            'R': smpl_R, # smpl global Rh Th # 
            'Th': Th,

            "params": params, # smpl params including R, Th
            'vertices': vertices, # world vertices
            
            # world space
            'img_all': img_all,
            'img_ray_d_all': img_ray_d_all,
            'msk_all': msk_all,
            'K_all': K_all,
            'R_all': R_all,
            'T_all': T_all,
            'rgb_all': rgb_all,
            'ray_o_all': ray_o_all,
            'ray_d_all': ray_d_all,
            'near_all': near_all,
            'far_all': far_all,
            'mask_at_box_all': mask_at_box_all,
            'bkgd_msk_all': bkgd_msk_all
        }


        return ret

    def __len__(self):
        return len(self.ims)


class H36MDatasetBatch(Dataset):
    def __init__(self, data_root, split='test', view_num=3, border=5, N_rand=1024*32, image_scaling=1.0, multi_person=0, num_instance=1,
                start=0, interval=10, poses_num=100, mean_shape=1, new_mask=0):
        super(H36MDatasetBatch, self).__init__()
        self.data_root = data_root
        self.split = split
        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']
        self.mean_shape = mean_shape
        self.new_mask = new_mask

        # # TODO
        # self.multi_view_demo = True
        # K, RT = render_utils.load_cam(ann_file)
        # render_w2c = render_utils.gen_path(RT, num_views=20)
        # self.render_w2c = render_w2c
        # self.K = K[0]

        if view_num==4:
            self.input_view =  [0,1,2,3] # [x for x in range(24)] # [0,1,2,3]
        elif view_num==3:
            self.input_view = [0,1,2] # [0,2,3]
        else:
            self.input_view =  [0,1,2,3]
        self.train_view = [0,1,2,3]
        self.test_view =  [0,1,2,3]
        self.output_view = self.train_view if split == 'train' else self.test_view

        self.i = start # start index 0
        self.i_intv = interval # interval 1
        self.ni = poses_num # number of used poses 30
        self.ims = np.array([
            np.array(ims_data['ims'])[self.output_view]
            for ims_data in annots['ims'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        ]) # image_name of all used images, shape (num of poses, num of output_views)

        self.nrays = N_rand
        self.border = border
        self.image_scaling = image_scaling
        self.num_instance = num_instance

        self.multi_person = multi_person
        human_dirs = [
            "./data/h36m/S1/Posing", "./data/h36m/S5/Posing", "./data/h36m/S6/Posing",
            "./data/h36m/S7/Posing", "./data/h36m/S9/Posing"#, "./data/h36m/S11/Posing"  "./data/h36m/S7/Posing"
            ]
        self.root_list = [data_root] if not multi_person else [x for x in human_dirs]
    

    def get_mask(self, index, view_index):
        msk_dir = 'refined_mask' if self.new_mask else 'mask_cihp'
        msk_path = os.path.join(self.data_root, msk_dir, self.ims[index][view_index])[:-4] + '.png'
        # msk_path = os.path.join(self.data_root, 'mask_cihp', self.ims[index][view_index])[:-4] + '.png'
        
        msk_cihp = imageio.imread(msk_path)
        # msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk_cihp = (msk_cihp > 200).astype(np.uint8) if self.new_mask else (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp if msk_cihp.shape[-1]!=3 else msk_cihp[...,0]
        o_msk_cihp = msk.copy()
        border = self.border
        kernel = np.ones((border, border), np.uint8)
        
        # msk_erode = cv2.erode(msk.copy(), kernel)
        # msk_dilate = cv2.dilate(msk.copy(), kernel)
        # msk[(msk_dilate - msk_erode) == 1] = 100
        
        # msk_erode = msk.copy()
        msk_erode = msk.copy() if self.new_mask else cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)

        kernel_ = np.ones((border+3, border+3), np.uint8)
        msk_dilate_ = cv2.dilate(msk.copy(), kernel_)

        msk[(msk_dilate - msk_erode) == 1] = 100
        msk[(msk_dilate_ - msk_dilate) == 1] = 200

        return msk, o_msk_cihp

    def prepare_input_t(self, t_vertices_path):
        vertices_path = t_vertices_path
        xyz = np.load(vertices_path).astype(np.float32)
        vertices = xyz

        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        
        feature = cxyz
        # feature = np.ones((6890,1)) * 0.01

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array([0.005, 0.005, 0.005])
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, bounds

    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, 'new_vertices', '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        vertices = xyz
        # nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate 
        params_path = os.path.join(self.data_root, "new_params", '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        # nxyz = nxyz.astype(np.float32)
        # feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)
        feature = cxyz

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array([0.005, 0.005, 0.005])
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, world_bounds, bounds, Rh, Th, vertices, params # , center, rot, trans

    def update(self, data_root):
        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']
        self.ims = np.array([
            np.array(ims_data['ims'])[self.output_view]
            for ims_data in annots['ims'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        ]) # image_name of all used images, shape (num of poses, num of output_views)
    
    def __getitem__(self, pose_index):
        
        data_root_i = np.random.randint(len(self.root_list)) if self.multi_person else 0
        self.data_root = self.root_list[data_root_i]
        self.update(self.data_root)
        pose_index = pose_index % self.ni
        # print(self.data_root)
        img_all, K_all, R_all, T_all, rgb_all, ray_o_all, ray_d_all, near_all, far_all, msk_all = [], [], [], [], [], [], [], [], [], []
        mask_at_box_all, bkgd_msk_all, img_ray_d_all, o_img_all,  msk_cihp_all = [], [], [], [], []
        
        for idx, view_index in enumerate(self.output_view):
        # TODO
        # for view_index, RT in enumerate(self.render_w2c):
            # idx = view_index
            # TODO
            # if view_index < 3:
            
            # Load image, mask, K, D, R, T
            img_path = os.path.join(self.data_root, self.ims[pose_index][idx].replace('\\', '/'))
            img = np.array(imageio.imread(img_path).astype(np.float32) / 255.)
            # msk = np.array(self.get_mask(pose_index, idx))
            msk, msk_cihp = self.get_mask(pose_index, idx)
            
            # img = cv2.resize(img, (1002, 1000), interpolation=cv2.INTER_AREA)
            # msk = cv2.resize(msk, (1002, 1000), interpolation=cv2.INTER_NEAREST)
            # msk_cihp = cv2.resize(msk_cihp, (1002, 1000), interpolation=cv2.INTER_NEAREST)

            img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (1000, 1000), interpolation=cv2.INTER_NEAREST)
            msk_cihp = cv2.resize(msk_cihp, (1000, 1000), interpolation=cv2.INTER_NEAREST)

            K = np.array(self.cams['K'][view_index])
            D = np.array(self.cams['D'][view_index])
            img = cv2.undistort(img, K, D)
            msk = cv2.undistort(msk, K, D)
            R = np.array(self.cams['R'][view_index])
            T = np.array(self.cams['T'][view_index]) / 1000.
        
            # Reduce the image resolution by ratio, then remove the back ground
            ratio = self.image_scaling
            if ratio != 1.:
                H, W = int(img.shape[0] * ratio), int(img.shape[1] * ratio)
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                mask_bkgd = True
                if mask_bkgd:
                    img[msk == 0] = 0
                K[:2] = K[:2] * ratio
            else:
                mask_bkgd = True
                if mask_bkgd:
                    img[msk == 0] = 0
                
            # # TODO
            # else:
            #     img_path = os.path.join(self.data_root, self.ims[pose_index][0].replace('\\', '/'))
            #     img = np.array(imageio.imread(img_path).astype(np.float32) / 255.)
            #     msk, msk_cihp = self.get_mask(pose_index, 0)
            #     # img = imageio.imread(img_path).astype(np.float32) / 255.
            #     # msk = self.get_mask(index, 0)
            
            # Prepare the smpl input, including the current pose and canonical pose
            if view_index == self.output_view[0]:
                i = int(os.path.basename(img_path)[:-4])
                # print(view_index, img_path)
                feature, coord, out_sh, world_bounds, bounds, Rh, Th, vertices, params = self.prepare_input(i)                
                if self.mean_shape:
                    path = os.path.join('n_X_template_tvertices.npy')
                else:
                    # path = os.path.join(self.data_root, 'tvertices.npy')
                    path = os.path.join(self.data_root, '45_big_pose_tvertices.npy')
                smpl_R = cv2.Rodrigues(Rh)[0].astype(np.float32)
                params["R"] = cv2.Rodrigues(params['Rh'])[0].astype(np.float32)
                t_vertices = np.load(path)
                t_feature, t_coord, t_out_sh, t_bounds = self.prepare_input_t(t_vertices_path=path)
            
            o_img = img.copy()
            o_img[msk_cihp==0] = 0
            
            # Sample rays in target space world coordinate
            rgb, ray_o, ray_d, near, far, coord_, mask_at_box, bkgd_msk,  img_ray_d = if_nerf_dutils.sample_ray_h36m_batch(
                    img, msk, K, R, T, world_bounds, self.nrays, self.split, ratio=0.6)
            
            # TODO
            # bkgd_msk,  img_ray_d = [], img
            # rgb, ray_o, ray_d, near, far, coord_, mask_at_box = sample_ray(
            #             img, msk, self.K, RT[:3, :3], RT[:3,3:], world_bounds, self.nrays, self.split)
            
            
            # Pack all inputs of all views
            img = np.transpose(img, (2,0,1))
            o_img = np.transpose(o_img, (2,0,1))
            img_ray_d = np.transpose(img_ray_d, (2,0,1))
            if view_index in self.input_view:
                img_all.append(img)
                img_ray_d_all.append(img_ray_d)
                K_all.append(K)
                R_all.append(R)
                T_all.append(T)
            o_img_all.append(o_img)
            msk_cihp_all.append(msk_cihp)
            msk_all.append(msk)
            rgb_all.append(rgb)
            ray_o_all.append(ray_o)
            ray_d_all.append(ray_d)
            near_all.append(near)
            far_all.append(far)
            mask_at_box_all.append(mask_at_box)
            bkgd_msk_all.append(bkgd_msk)

        img_all = np.stack(img_all, axis=0)
        o_img_all = np.stack(o_img_all, axis=0)
        msk_cihp_all = np.stack(msk_cihp_all, axis=0)
        img_ray_d_all = np.stack(img_ray_d_all, axis=0)
        msk_all = np.stack(msk_all, axis=0)
        K_all = np.stack(K_all, axis=0)
        R_all = np.stack(R_all, axis=0)
        T_all = np.stack(T_all, axis=0)
        # if self.split == "train":
        rgb_all = np.stack(rgb_all, axis=0)
        ray_o_all = np.stack(ray_o_all, axis=0)
        ray_d_all = np.stack(ray_d_all, axis=0)
        near_all = np.stack(near_all, axis=0)[...,None]
        far_all = np.stack(far_all, axis=0)[...,None]
        mask_at_box_all = np.stack(mask_at_box_all, axis=0)
        bkgd_msk_all = np.stack(bkgd_msk_all, axis=0)
        

        ret = {
            'pose_index': pose_index,
            "instance_idx": data_root_i, 
            'R': smpl_R, # smpl global Rh Th # 
            'Th': Th,
            'gender': 2,

            "params": params, # smpl params including R, Th
            'vertices': vertices, # world vertices
            'feature': feature, # smpl pose space xyz
            'coord': coord,    # smpl pose space coords
            'bounds': bounds, # smpl pose space bounds
            'out_sh': out_sh,   # smpl pose space out_sh

            't_vertices': t_vertices,
            't_feature': t_feature, # smpl pose space xyz
            't_coord': t_coord,    # smpl pose space coords
            't_bounds': t_bounds, # smpl pose space bounds
            't_out_sh': t_out_sh,   # smpl pose space out_sh
            
            # world space
            'img_all': img_all,
            'o_img_all': o_img_all,
            'img_ray_d_all': img_ray_d_all,
            'msk_cihp_all': msk_cihp_all,
            'msk_all': msk_all,
            'K_all': K_all,
            'R_all': R_all,
            'T_all': T_all,
            'rgb_all': rgb_all,
            'ray_o_all': ray_o_all,
            'ray_d_all': ray_d_all,
            'near_all': near_all,
            'far_all': far_all,
            'mask_at_box_all': mask_at_box_all,
            'bkgd_msk_all': bkgd_msk_all
        }


        return ret

    def __len__(self):
        return len(self.ims) * self.num_instance


class H36MDatasetPair(Dataset):
    def __init__(self, data_root, split='test', view_num=3, border=5, N_rand=1024*32, image_scaling=1.0, multi_person=0, num_instance=1,
                start=0, interval=10, poses_num=100, random_pair=0, mean_shape=1, new_mask=0, test_persons=2):
        super(H36MDatasetPair, self).__init__()
        self.data_root = data_root
        self.split = split
        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams'] # K:[(4*4), (4*4), (4*4), (4*4)], D, R, T
        self.mean_shape = mean_shape
        self.new_mask = new_mask

        if view_num==4:
            self.input_view =  [0,1,2,3] # [x for x in range(24)] # [0,1,2,3]
        elif view_num==3:
            self.input_view = [0,1,2] # [0,2,3]
        else:
            self.input_view =  [0,1,2,3]
        self.train_view = [0,1,2,3] if multi_person else [0,1,2]
        self.test_view =  [0,1,2,3]
        self.output_view = self.train_view if split == 'train' else self.test_view

        self.i = start # start index 0
        self.i_intv = interval # interval 1
        self.ni = poses_num # number of used poses 30
        self.ims = np.array([
            np.array(ims_data['ims'])[self.output_view]
            for ims_data in annots['ims'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        ]) # image_name of all used images, shape (num of poses, num of output_views)

        self.nrays = N_rand
        self.border = border
        # self.pairs = np.loadtxt('training_set_100_pose_same_pair.txt').reshape(-1,2)
        self.random_num = 5
        self.pairs = np.random.randint(0,self.ni,(self.ni, self.random_num))
        self.img_scaling = image_scaling
        self.num_instance = num_instance

        self.random_pair = random_pair
        self.multi_person = multi_person
        
        # if test_persons==2:
        #     human_dirs = ['data/h36m/S1/Posing', 'data/h36m/S5/Posing', 'data/h36m/S6/Posing', 'data/h36m/S7/Posing', 'data/h36m/S9/Posing'] # , 'data/h36m/S11/Posing' 'msrah36m/S7'
        # elif test_persons==1: 
        #     human_dirs = ['data/h36m/S1/Posing', 'data/h36m/S5/Posing', 'data/h36m/S6/Posing', 'data/h36m/S8/Posing', 'data/h36m/S11/Posing'] 
        # else:
        #     human_dirs = ['data/h36m/S7/Posing', 'data/h36m/S8/Posing', 'data/h36m/S9/Posing', 'data/h36m/S11/Posing'] 
        human_dirs = ['data/h36m/S1/Posing', 'data/h36m/S5/Posing', 'data/h36m/S6/Posing', 'data/h36m/S7/Posing', 'data/h36m/S8/Posing', 'data/h36m/S9/Posing', 'data/h36m/S11/Posing'] # , 'data/h36m/S11/Posing' 'msrah36m/S7'
        test_person = human_dirs.pop(int(test_persons))

        self.root_list = [data_root] if not multi_person else [x for x in human_dirs]
        print(self.root_list)

    def update(self, data_root):
        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams'] # K:[(4*4), (4*4), (4*4), (4*4)], D, R, T
        self.ims = np.array([
            np.array(ims_data['ims'])[self.output_view]
            for ims_data in annots['ims'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        ])# image_name array, shape (150, 4)
        
    def get_mask(self, index, view_index):
        
        msk_dir = 'refined_mask' if self.new_mask else 'mask_cihp'
        msk_path = os.path.join(self.data_root, msk_dir, self.ims[index][view_index])[:-4] + '.png'
        # msk_path = os.path.join(self.data_root, 'mask_cihp', self.ims[index][view_index])[:-4] + '.png'
        
        msk_cihp = imageio.imread(msk_path)
        # if len(np.unique(msk_cihp)) > 3: 
        #     print(msk_path, len(np.unique(msk_cihp)))
        msk_cihp = (msk_cihp > 200).astype(np.uint8) if self.new_mask else (msk_cihp != 0).astype(np.uint8)
        # msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp if msk_cihp.shape[-1]!=3 else msk_cihp[...,0]
        o_msk_cihp = msk.copy()
        border = self.border
        kernel = np.ones((border, border), np.uint8)
        
        # msk_erode = cv2.erode(msk.copy(), kernel)
        # msk_dilate = cv2.dilate(msk.copy(), kernel)
        # msk[(msk_dilate - msk_erode) == 1] = 100
        
        msk_erode = msk.copy() if self.new_mask else cv2.erode(msk.copy(), kernel)  # two idff: 1. this line 2. border 5->3
        msk_dilate = cv2.dilate(msk.copy(), kernel)

        kernel_ = np.ones((border+3, border+3), np.uint8)
        msk_dilate_ = cv2.dilate(msk.copy(), kernel_)

        msk[(msk_dilate - msk_erode) == 1] = 100
        msk[(msk_dilate_ - msk_dilate) == 1] = 200

        return msk, o_msk_cihp

    def prepare_input_t(self, t_vertices_path):
        vertices_path = t_vertices_path
        xyz = np.load(vertices_path).astype(np.float32)
        vertices = xyz

        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        feature = cxyz
        # feature = np.ones((6890,1)) * 0.01

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array([0.005, 0.005, 0.005])
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, bounds

    def prepare_input(self, i):
        # read vetices from the npy file
        vertices_path = os.path.join(self.data_root, 'new_vertices', '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        vertices = xyz

        # obtain the original bounds for this human pose
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate 
        params_path = os.path.join(self.data_root, "new_params", '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        # construct point feature as xyz
        cxyz = xyz.astype(np.float32)
        feature = cxyz

        # construct the coordinate in smpl space
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array([0.005, 0.005, 0.005])
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, world_bounds, bounds, Rh, Th, vertices, params # , center, rot, trans

    def __getitem__(self, pose_index):
        data_root_i = np.random.randint(len(self.root_list)) if self.multi_person else 0
        self.data_root = self.root_list[data_root_i]
        self.update(self.data_root)
        pose_index = pose_index % self.ni
        if self.random_pair:
            # pair = [np.random.randint(0,self.ni), np.random.randint(0,self.ni)]
            thre = np.random.uniform(low=0.0, high=1.0)
            pair = [pose_index, pose_index if thre<0.5 else np.random.randint(pose_index,len(self.ims))]
            # pair = [pose_index, pose_index if thre<0.5 else np.random.randint(0,len(self.ims))]
            
            # other_index = np.random.randint(0,self.random_num + 1)
            # choice_list = self.pairs[pose_index].tolist()
            # choice_list.append(pose_index)
            # if (pose_index % 5)==0:
            #     tmp=choice_list[other_index]
            # else:
            #     tmp=pose_index
            # pair = [pose_index, tmp]
                
            # pair = [pose_index, choice_list[other_index]]
        
        sp_tp_ret = {}
        
        # print(self.data_root, pair)
        for k, pose_index in enumerate([int(pair[0]), int(pair[1])]):
            # load 4 views images path for pose index(index of 150 selected poses)
            img_path_list = self.ims[pose_index]
            all_img_path = []
            for i, path in enumerate(img_path_list):
                all_img_path.append(os.path.join(self.data_root, path))
            img_all, K_all, R_all, T_all, rgb_all, ray_o_all, ray_d_all, near_all, far_all, msk_all = [], [], [], [], [], [], [], [], [], []
            mask_at_box_all, bkgd_msk_all, img_ray_d_all, o_img_all, msk_cihp_all = [], [], [], [], []
            # print(all_img_path)
            for view_index in self.output_view:
                # load and resize
                img_path = all_img_path[view_index]
                img = imageio.imread(img_path).astype(np.float32) / 255.
                msk, msk_cihp = self.get_mask(pose_index, view_index)
                img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_AREA)
                msk = cv2.resize(msk, (1000, 1000), interpolation=cv2.INTER_NEAREST)
                msk_cihp = cv2.resize(msk_cihp, (1000, 1000), interpolation=cv2.INTER_NEAREST)

                # load intrinsic parameters and extrinsic paramters for each view
                K = np.array(self.cams['K'][view_index])
                D = np.array(self.cams['D'][view_index])
                img = cv2.undistort(img, K, D)
                msk = cv2.undistort(msk, K, D)
                R = np.array(self.cams['R'][view_index])
                T = np.array(self.cams['T'][view_index]) / 1000.

                # reduce the image resolution by ratio, mask the bkgd
                ratio = self.img_scaling
                if ratio != 1.:
                    H, W = int(img.shape[0] * ratio), int(img.shape[1] * ratio)
                    K[:2] = K[:2] * ratio
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                    msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                    img[msk == 0] = 0
                else:
                    img[msk == 0] = 0
                                
                # load and process smpl related data
                if view_index == 0:
                    i = int(os.path.basename(img_path)[:-4])
                    feature, coord, out_sh, world_bounds, bounds, Rh, Th, vertices, params = self.prepare_input(i)                
                    if self.mean_shape:
                        path = os.path.join('n_X_template_tvertices.npy')
                    else:
                        path = os.path.join(self.data_root, '45_big_pose_tvertices.npy')
                    t_vertices = np.load(path)
                    smpl_R = cv2.Rodrigues(Rh)[0].astype(np.float32)
                    params["R"] = cv2.Rodrigues(params['Rh'])[0].astype(np.float32)
                    t_feature, t_coord, t_out_sh, t_bounds = self.prepare_input_t(t_vertices_path=path)

                o_img = img.copy()
                o_img[msk_cihp==0] = 0
                # sample rays for each view
                # rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray_h36m(
                rgb, ray_o, ray_d, near, far, coord_, mask_at_box, bkgd_msk,  img_ray_d = if_nerf_dutils.sample_ray_h36m_batch(
                        img, msk, K, R, T, world_bounds, self.nrays, self.split, ratio=0.6, img_path=img_path)
                                
                img = np.transpose(img, (2,0,1))
                o_img = np.transpose(o_img, (2,0,1))
                img_ray_d = np.transpose(img_ray_d, (2,0,1))

                if view_index in self.input_view:
                    img_all.append(img)
                    img_ray_d_all.append(img_ray_d)
                    K_all.append(K)
                    R_all.append(R)
                    T_all.append(T)
                o_img_all.append(o_img)
                msk_cihp_all.append(msk_cihp)
                msk_all.append(msk)
                rgb_all.append(rgb)
                ray_o_all.append(ray_o)
                ray_d_all.append(ray_d)
                near_all.append(near)
                far_all.append(far)
                mask_at_box_all.append(mask_at_box)
                bkgd_msk_all.append(bkgd_msk)

            img_all = np.stack(img_all, axis=0)
            o_img_all = np.stack(o_img_all, axis=0)
            img_ray_d_all = np.stack(img_ray_d_all, axis=0)
            msk_cihp_all = np.stack(msk_cihp_all, axis=0)
            msk_all = np.stack(msk_all, axis=0)
            K_all = np.stack(K_all, axis=0)
            R_all = np.stack(R_all, axis=0)
            T_all = np.stack(T_all, axis=0)
            # if self.split == "train":
            rgb_all = np.stack(rgb_all, axis=0)
            ray_o_all = np.stack(ray_o_all, axis=0)
            ray_d_all = np.stack(ray_d_all, axis=0)
            near_all = np.stack(near_all, axis=0)[...,None]
            far_all = np.stack(far_all, axis=0)[...,None]
            mask_at_box_all = np.stack(mask_at_box_all, axis=0)
            bkgd_msk_all = np.stack(bkgd_msk_all, axis=0)

            ret = {
                'pose_index': pose_index,
                "instance_idx": data_root_i, 
                'R': smpl_R, # smpl global Rh Th # 
                'Th': Th,
                'gender': 2,

                "params": params, # smpl params including R, Th
                'vertices': vertices, # world vertices
                'feature': feature, # smpl pose space xyz
                'coord': coord,    # smpl pose space coords
                'bounds': bounds, # smpl pose space bounds
                'out_sh': out_sh,   # smpl pose space out_sh

                't_vertices': t_vertices,
                't_feature': t_feature, # smpl pose space xyz
                't_coord': t_coord,    # smpl pose space coords
                't_bounds': t_bounds, # smpl pose space bounds
                't_out_sh': t_out_sh,   # smpl pose space out_sh
                
                # world space
                'img_all': img_all,
                'o_img_all': o_img_all,
                'img_ray_d_all': img_ray_d_all,
                'msk_cihp_all': msk_cihp_all,
                'msk_all': msk_all,
                'K_all': K_all,
                'R_all': R_all,
                'T_all': T_all,
                'rgb_all': rgb_all,
                'ray_o_all': ray_o_all,
                'ray_d_all': ray_d_all,
                'near_all': near_all,
                'far_all': far_all,
                'mask_at_box_all': mask_at_box_all,
                'bkgd_msk_all': bkgd_msk_all
            }

            if k==0:
                sp_tp_ret['sp_input'] = ret
            else:
                sp_tp_ret['tp_input'] = ret
        
        return sp_tp_ret

    def __len__(self):
        return len(self.ims) * self.num_instance


class H36MDatasetBatchAll(Dataset):
    def __init__(self, data_root, split='test', view_num=3, border=5, N_rand=1024*32, image_scaling=1.0, multi_person=0, num_instance=1,
                start=0, interval=10, poses_num=100, mean_shape=1):
        super(H36MDatasetBatchAll, self).__init__()
        self.data_root = data_root
        self.split = split
        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']
        self.mean_shape = mean_shape

        if view_num==4:
            self.input_view =  [0,1,2,3] # [x for x in range(24)] # [0,1,2,3]
        elif view_num==3:
            self.input_view = [0,2,3]
        else:
            self.input_view =  [0,1,2,3]
        self.train_view = [0,1,2,3]
        self.test_view =  [0,1,2,3]
        self.output_view = self.train_view if split == 'train' else self.test_view

        self.multi_person = multi_person
        human_dirs = [
            "./data/h36m/S5/Posing", "./data/h36m/S6/Posing", "./data/h36m/S7/Posing", 
            "./data/h36m/S8/Posing", "./data/h36m/S9/Posing", "./data/h36m/S1/Posing" # , "./data/h36m/S1/Posing" #./data/h36m/S11/Posing
            ]
        self.root_list = [data_root] if not multi_person else [x for x in human_dirs]


        # S1:990   S5:1880   S6:1160    S7:2920	   S8:1680	  S9:1960	S11:1400
        self.poses_num_dict = {
            "./data/h36m/S1/Posing":99,  "./data/h36m/S5/Posing":188, "./data/h36m/S6/Posing":116, "./data/h36m/S7/Posing":292,
            "./data/h36m/S8/Posing":168, "./data/h36m/S9/Posing":196, "./data/h36m/S11/Posing":140
        }
        self.poses_num_list = np.array([self.poses_num_dict[root] for root in self.root_list])
        self.acc_posed_num_list = self.poses_num_list.cumsum()


        self.i = start # start index 0
        self.i_intv = interval # interval 1
        self.ni = poses_num # number of used poses 30
        self.ims = np.array([
            np.array(ims_data['ims'])[self.output_view]
            for ims_data in annots['ims'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        ]) # image_name of all used images, shape (num of poses, num of output_views)

        self.nrays = N_rand
        self.border = border
        self.image_scaling = image_scaling
        self.num_instance = num_instance

    
    def get_mask(self, index, view_index):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[index][view_index])[:-4] + '.png'
        msk_cihp = imageio.imread(msk_path)
        msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp

        border = self.border
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)

        kernel_ = np.ones((border+3, border+3), np.uint8)
        msk_dilate_ = cv2.dilate(msk.copy(), kernel_)

        msk[(msk_dilate - msk_erode) == 1] = 100
        msk[(msk_dilate_ - msk_dilate) == 1] = 200

        return msk

    def prepare_input_t(self, t_vertices_path):
        vertices_path = t_vertices_path
        xyz = np.load(vertices_path).astype(np.float32)
        vertices = xyz

        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        
        feature = cxyz
        # feature = np.ones((6890,1)) * 0.01

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array([0.005, 0.005, 0.005])
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, bounds

    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, 'new_vertices', '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        vertices = xyz
        # nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate 
        params_path = os.path.join(self.data_root, "new_params", '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        # nxyz = nxyz.astype(np.float32)
        # feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)
        feature = cxyz

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array([0.005, 0.005, 0.005])
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, world_bounds, bounds, Rh, Th, vertices, params # , center, rot, trans

    def update(self, data_root):
        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']
        self.ims = np.array([
            np.array(ims_data['ims'])[self.output_view]
            for ims_data in annots['ims'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        ]) # image_name of all used images, shape (num of poses, num of output_views)
    
    def __getitem__(self, pose_index):
        
        # data_root_i = np.random.randint(len(self.root_list)) if self.multi_person else 0
        # self.data_root = self.root_list[data_root_i]
        # self.update(self.data_root)
        # pose_index = pose_index % self.ni

        data_root_i = np.where(self.acc_posed_num_list>pose_index)[0].min()
        pose_index = pose_index - self.acc_posed_num_list[data_root_i-1] if data_root_i!= 0 else 0
        self.ni = self.poses_num_list[data_root_i]
        self.data_root = self.root_list[data_root_i]
        self.update(self.data_root)
        
        # print(self.data_root)
        img_all, K_all, R_all, T_all, rgb_all, ray_o_all, ray_d_all, near_all, far_all, msk_all = [], [], [], [], [], [], [], [], [], []
        mask_at_box_all, bkgd_msk_all, img_ray_d_all = [], [], []

        for idx, view_index in enumerate(self.output_view):
            # Load image, mask, K, D, R, T
            img_path = os.path.join(self.data_root, self.ims[pose_index][idx].replace('\\', '/'))
            img = np.array(imageio.imread(img_path).astype(np.float32) / 255.)
            # print(img_path)
            msk = np.array(self.get_mask(pose_index, idx))
            img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (1000, 1000), interpolation=cv2.INTER_NEAREST)

            K = np.array(self.cams['K'][view_index])
            D = np.array(self.cams['D'][view_index])
            img = cv2.undistort(img, K, D)
            msk = cv2.undistort(msk, K, D)
            R = np.array(self.cams['R'][view_index])
            T = np.array(self.cams['T'][view_index]) / 1000.

            # Reduce the image resolution by ratio, then remove the back ground
            ratio = self.image_scaling
            if ratio != 1.:
                H, W = int(img.shape[0] * ratio), int(img.shape[1] * ratio)
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                mask_bkgd = True
                if mask_bkgd:
                    img[msk == 0] = 0
                K[:2] = K[:2] * ratio
            else:
                mask_bkgd = True
                if mask_bkgd:
                    img[msk == 0] = 0
            
            # Prepare the smpl input, including the current pose and canonical pose
            if view_index == self.output_view[0]:
                i = int(os.path.basename(img_path)[:-4])
                feature, coord, out_sh, world_bounds, bounds, Rh, Th, vertices, params = self.prepare_input(i)                
                # path = os.path.join(self.data_root, '45_big_pose_tvertices.npy') # before_200_45_big_pose_tvertices # 45_big_pose_tvertices # posing_tvertices
                path = os.path.join('n_X_template_tvertices.npy') # before_200_45_big_pose_tvertices # 45_big_pose_tvertices # posing_tvertices
                smpl_R = cv2.Rodrigues(Rh)[0].astype(np.float32)
                params["R"] = cv2.Rodrigues(params['Rh'])[0].astype(np.float32)
                t_vertices = np.load(path)
                t_feature, t_coord, t_out_sh, t_bounds = self.prepare_input_t(t_vertices_path=path)
            
            # Sample rays in target space world coordinate
            rgb, ray_o, ray_d, near, far, coord_, mask_at_box, bkgd_msk,  img_ray_d = if_nerf_dutils.sample_ray_h36m_batch(
                    img, msk, K, R, T, world_bounds, self.nrays, self.split, ratio=0.6)

            # Pack all inputs of all views
            img = np.transpose(img, (2,0,1))
            img_ray_d = np.transpose(img_ray_d, (2,0,1))
            if view_index in self.input_view:
                img_all.append(img)
                img_ray_d_all.append(img_ray_d)
                K_all.append(K)
                R_all.append(R)
                T_all.append(T)
            msk_all.append(msk)
            rgb_all.append(rgb)
            ray_o_all.append(ray_o)
            ray_d_all.append(ray_d)
            near_all.append(near)
            far_all.append(far)
            mask_at_box_all.append(mask_at_box)
            bkgd_msk_all.append(bkgd_msk)

        img_all = np.stack(img_all, axis=0)
        img_ray_d_all = np.stack(img_ray_d_all, axis=0)
        msk_all = np.stack(msk_all, axis=0)
        K_all = np.stack(K_all, axis=0)
        R_all = np.stack(R_all, axis=0)
        T_all = np.stack(T_all, axis=0)
        # if self.split == "train":
        rgb_all = np.stack(rgb_all, axis=0)
        ray_o_all = np.stack(ray_o_all, axis=0)
        ray_d_all = np.stack(ray_d_all, axis=0)
        near_all = np.stack(near_all, axis=0)[...,None]
        far_all = np.stack(far_all, axis=0)[...,None]
        mask_at_box_all = np.stack(mask_at_box_all, axis=0)
        bkgd_msk_all = np.stack(bkgd_msk_all, axis=0)
        

        ret = {
            'pose_index': pose_index,
            "instance_idx": data_root_i, 
            'R': smpl_R, # smpl global Rh Th # 
            'Th': Th,

            "params": params, # smpl params including R, Th
            'vertices': vertices, # world vertices
            'feature': feature, # smpl pose space xyz
            'coord': coord,    # smpl pose space coords
            'bounds': bounds, # smpl pose space bounds
            'out_sh': out_sh,   # smpl pose space out_sh

            't_vertices': t_vertices,
            't_feature': t_feature, # smpl pose space xyz
            't_coord': t_coord,    # smpl pose space coords
            't_bounds': t_bounds, # smpl pose space bounds
            't_out_sh': t_out_sh,   # smpl pose space out_sh
            
            # world space
            'img_all': img_all,
            'img_ray_d_all': img_ray_d_all,
            'msk_all': msk_all,
            'K_all': K_all,
            'R_all': R_all,
            'T_all': T_all,
            'rgb_all': rgb_all,
            'ray_o_all': ray_o_all,
            'ray_d_all': ray_d_all,
            'near_all': near_all,
            'far_all': far_all,
            'mask_at_box_all': mask_at_box_all,
            'bkgd_msk_all': bkgd_msk_all
        }


        return ret

    def __len__(self):
        # return len(self.ims) * self.num_instance
        return self.acc_posed_num_list[-1] # * self.output_view


def sample_ray(img, msk, K, R, T, bounds, nrays, split):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)
    pose = np.concatenate([R, T], axis=1)
    
    # bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)
    bound_mask = np.ones_like(msk)
    mask_bkgd = True

    # msk = msk * bound_mask
    bound_mask[msk == 100] = 0
    bound_mask[msk == 200] = 0
    
    if mask_bkgd:
        img[bound_mask != 1] = 0
    # imageio.imwrite("./objs/tmp.png", img*255)
    # imageio.imwrite("./objs/bound_mask.png", bound_mask*255)

    if split == 'train':
        
        nsampled_rays = 0
        body_sample_ratio = 0.5
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []

        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            n_rand = (nrays - nsampled_rays) - n_body

            # sample rays on body
            coord_body = np.argwhere(msk == 1)
            index = np.random.randint(0, len(coord_body), n_body)
            # index.sort()
            coord_body = coord_body[index]
            # sample rays in the bound mask
            coord = np.argwhere(bound_mask == 1)
            index2 = np.random.randint(0, len(coord), n_rand)
            # index2.sort()
            coord = coord[index2]
            coord = np.concatenate([coord_body, coord], axis=0)
            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]
            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord[mask_at_box])
            mask_at_box_list.append(mask_at_box[mask_at_box])
            nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)
        
    else:
        # rgb = img.reshape(-1, 3).astype(np.float32)
        # ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        # ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        # near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        # near = near.astype(np.float32)
        # far = far.astype(np.float32)
        # rgb = rgb[mask_at_box]
        # ray_o = ray_o[mask_at_box]
        # ray_d = ray_d[mask_at_box]
        # coord = np.zeros([len(rgb), 2]).astype(np.int64)

        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)

        near_all = np.zeros_like(ray_o[:,0])
        far_all = np.ones_like(ray_o[:,0])
        near_all[mask_at_box] = near 
        far_all[mask_at_box] = far 
        near = near_all
        far = far_all
        coord = np.zeros([len(rgb), 2]).astype(np.int64) # no use
        bkgd_msk = np.ones_like(msk) # no use

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box



# def main():
#     data_root = 'data/h36m/S9/Posing'
#     train_data  = Dataset(data_root)
#     trainloader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=10)
#     for i, data in enumerate(trainloader):
#         print(i)


# if __name__ == "__main__":
#     main()
# def main():
#     data_root = 'data/h36m/S9/Posing'
#     train_data  = Dataset(data_root)
#     trainloader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=10)
#     for i, data in enumerate(trainloader):
#         print(i)


# if __name__ == "__main__":
#     main()