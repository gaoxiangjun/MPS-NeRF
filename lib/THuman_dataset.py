from torch.utils.data import DataLoader, dataset
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import imageio
import cv2
import time
import lib.if_nerf_data_utils as if_nerf_dutils


class THumanDataset(Dataset):
    def __init__(self, data_root, split='test', view_num=24, N_rand=1024*32, multi_person=False, num_instance=1, 
                    start=0, interval=1, poses_num=30, image_scaling=1.0):
        super(THumanDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']
        self.nrays = N_rand
        

        self.train_view = [0,6,12,18] if view_num==4 else [x for x in range(view_num)] # [0,6,12,18]
        self.test_view =  [x for x in range(24)] # [x for x in range(view_num)] # [0,6,12,18]
        self.input_view = [0,6,12,18] if view_num==4 else [0,1, 6,7, 12,13, 18,19] # [0,6,12,18] # [0,1, 6,7, 12,13, 18,19]
        self.output_view = self.train_view if split == 'train' else self.test_view
        
        """
        annots = {
            'cams':{
                'K':[],#N arrays, (3, 3)
                'D':[],#(5, 1), all zeros
                'R':[],#(3, 3)
                'T':[] #(3, 1)
            },

            'ims':[
                # {'ims':['54138969/000000.jpg', '55011271/000000.jpg''60457274/000000.jpg']}, # same pose different views
                # {'ims':[]} 
                #  repeat time is number of poses
            ]
        }
        """
        self.i = start # start index
        self.i_intv = interval # interval
        self.ni = poses_num # number of used poses
        self.ims = np.array([
            np.array(ims_data['ims'])[self.output_view]
            for ims_data in annots['ims'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        ]) # image_name of all used images, shape (num of poses, num of output_views)
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[self.output_view]
            for ims_data in annots['ims'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        ]) # view index of output_view, shape (num of poses, num of output_views)

        self.multi_person = multi_person
        all_human_data_root = os.path.join(os.path.dirname(data_root))
        self.human_list = os.path.join(all_human_data_root, 'data/THuman_1_human_list.txt')# if split == 'train' else 'THuman/test_human_list.txt'
        with open(self.human_list) as f:
            human_dirs = f.readlines()[0:num_instance]
        exclude_dirs = ['results_gyx_20181012_ym_2_F', 'results_gyx_20181013_lqj_1_F', 'results_gyx_20181013_yry_1_F', 'results_gyx_20181013_znn_1_F', 
                        'results_gyx_20181013_zsh_1_M', 'results_gyx_20181013_zyj_1_F', 'results_gyx_20181014_sxy_2_F', 'results_gyx_20181015_dc_2_F', 
                        'results_gyx_20181015_gh_2_F'] # pose number is less than 30
        # dataset_root = '../THuman_1/nerf_data/'
        self.root_list = [data_root] if not multi_person else [os.path.join(all_human_data_root, x.strip()) for x in human_dirs if x.strip() not in exclude_dirs]
        self.image_scaling = image_scaling

    def update(self, data_root):
        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']
        self.ims = np.array([
            np.array(ims_data['ims'])[self.output_view]
            for ims_data in annots['ims'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        ]) # image_name of all used images, shape (num of poses, num of output_views)
    
    def get_mask(self, pose_index, idx):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[pose_index][idx].replace('\\', '/').replace('jpg', 'png'))
        msk = imageio.imread(msk_path)
        msk[msk!=0]=255
        return msk

    def prepare_input_t(self, t_vertices_path):
        vertices_path = t_vertices_path
        xyz = np.load(vertices_path).astype(np.float32)

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

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate 
        params_path = os.path.join(self.data_root, "new_params", '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        R = params['R']
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        # construct point feature
        cxyz = xyz.astype(np.float32)
        # nxyz = np.zeros_like(xyz).astype(np.float32)
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

        return feature, coord, out_sh, world_bounds, bounds, vertices, params # , center, rot, trans

    def __getitem__(self, pose_index):
        """
            pose_index: [0, number of used poses), the pose index of selected poses
            view_index: [0, number of all view)
            mask_at_box_all: mask of 2D bounding box which is the 3D bounding box projection 
                training: all one array, not fixed length, no use for training
                Test: zero and one array, fixed length which can be reshape to (H,W)
            bkgd_msk_all: mask of foreground and background
                trainning: for acc loss
                test: no use
        """
        data_root_i = np.random.randint(len(self.root_list)) if self.multi_person else 0
        self.data_root = self.root_list[data_root_i]
        self.update(self.data_root)

        img_all, K_all, R_all, T_all, rgb_all, ray_o_all, ray_d_all, near_all, far_all, msk_all = [], [], [], [], [], [], [], [], [], []
        mask_at_box_all, bkgd_msk_all, img_ray_d_all = [], [], []

        for idx, view_index in enumerate(self.output_view):
            # Load image, mask, K, D, R, T
            img_path = os.path.join(self.data_root, self.ims[pose_index][idx].replace('\\', '/'))
            img = np.array(imageio.imread(img_path).astype(np.float32) / 255.)
            msk = np.array(self.get_mask(pose_index, idx)) / 255.
            img[msk == 0] = 0
            K = np.array(self.cams['K'][view_index])
            D = np.array(self.cams['D'][view_index])
            img = cv2.undistort(img, K, D)
            msk = cv2.undistort(msk, K, D)
            R = np.array(self.cams['R'][view_index])
            T = np.array(self.cams['T'][view_index])
            
            # reduce the image resolution by ratio, mask the bkgd
            ratio = self.image_scaling
            if ratio != 1.:
                H, W = int(img.shape[0] * ratio), int(img.shape[1] * ratio)
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                K[:2] = K[:2] * ratio

            # Prepare the smpl input, including the current pose and canonical pose
            if view_index == self.output_view[0]:
                # i: the pose index of all poses this person has, not the pose index of getitem input
                i = int(os.path.basename(img_path)[:-4])
                feature, coord, out_sh, world_bounds, bounds, vertices, params = self.prepare_input(i)
                path = os.path.join('45_big_pose_tvertices_mean.npy')
                # path = os.path.join(self.data_root, '45_big_pose_tvertices.npy')
                t_vertices = np.load(path)
                t_feature, t_coord, t_out_sh, t_bounds = self.prepare_input_t(t_vertices_path=path)
            
            # Sample rays in target space world coordinate
            rgb, ray_o, ray_d, near, far, coord_, mask_at_box, bkgd_msk, img_ray_d = if_nerf_dutils.sample_ray_h36m(
                    img, msk, K, R, T, world_bounds, self.nrays, self.split)

            # Pack all inputs of all views
            img = np.transpose(img, (2,0,1))
            img_ray_d = np.transpose(img_ray_d, (2,0,1))
            if view_index in self.input_view:
                img_all.append(img)
                img_ray_d_all.append(img_ray_d)
                K_all.append(K)
                R_all.append(R)
                T_all.append(T)
            msk_all.append(msk) # For test, show gt image
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
            'pose_index': pose_index, # pose_index in selected poses
            "params": params, # smpl params including smpl global R, Th
            "instance_idx": data_root_i, # person instance idx

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
        return len(self.ims)


class THumanDatasetBatch(Dataset):
    def __init__(self, data_root, split='test', view_num=24, N_rand=1024*32, multi_person=False, num_instance=1, start=0, interval=1, poses_num=30, 
                image_scaling=1.0, male=0, mean_shape=0, model=None, finetune=False):
        super(THumanDatasetBatch, self).__init__()
        self.data_root = data_root
        self.split = split
        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']
        self.nrays = N_rand
        self.mean_shape=mean_shape

        # if view_num==24:
        #     self.train_view = [x for x in range(24)] # [x for x in range(24)] # [0,1,2,3]
        # else:
        #     self.train_view = [0,1, 6,7, 12,13, 18,19] # [0,1,2,3] # [x for x in range(20)]
        # self.input_view =  [0,1, 6,7, 12,13, 18,19] # [0,1,2,3] # [0,1, 6,7, 12,13, 18,19]
        if view_num==8:
            self.input_view =  [0,3,6,9,12,15,18,21] # [x for x in range(24)] # [0,1,2,3]
        elif view_num==6:
            self.input_view = [0,4,8,12,16,20] # [0,1,2,3] # [x for x in range(20)]
        elif view_num==4:
            self.input_view = [0,6,12,18] # [0, 4, 12, 16] # [0,6,12,18] [0,3,12,15] [4,12,20]
        elif view_num==3:
            self.input_view = [4, 12, 20]
        elif view_num==12:
            self.input_view = [0,2,4,6,8,10,12,14,16,18,20,22]
         # [0,1,2,3] # [0,1, 6,7, 12,13, 18,19]
        self.train_view = self.input_view if (model=="ani_nerf" or finetune) else [x for x in range(24)]
        # self.train_view = [x for x in range(24)]
        self.test_view =  [x for x in range(24)] # [0,6,12,18]#
        self.output_view = self.train_view if split == 'train' else self.test_view
        print("output view: ", self.output_view)
        """
        annots = {
            'cams':{
                'K':[],#N arrays, (3, 3)
                'D':[],#(5, 1), all zeros
                'R':[],#(3, 3)
                'T':[] #(3, 1)
            },

            'ims':[
                # {'ims':['54138969/000000.jpg', '55011271/000000.jpg''60457274/000000.jpg']}, # same pose different views
                # {'ims':[]} 
                #  repeat time is number of poses
            ]
        }
        """
        self.i = start # start index 0
        self.i_intv = interval # interval 1
        self.ni = poses_num # number of used poses 30
        self.ims = np.array([
            np.array(ims_data['ims'])[self.output_view]
            for ims_data in annots['ims'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        ]) # image_name of all used images, shape (num of poses, num of output_views)
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[self.output_view]
            for ims_data in annots['ims'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        ]) # view index of output_view, shape (num of poses, num of output_views)

        self.multi_person = multi_person
        self.num_instance = num_instance
        all_human_data_root = os.path.join(os.path.dirname(data_root))
        self.human_list = os.path.join('data/THuman_1_male_list.txt') if male==1 else os.path.join('data/THuman_1_human_list.txt')
        # self.human_list = os.path.join('THuman_1_human_list.txt')     
        # self.human_list = os.path.join(all_human_data_root, 'THuman_1_human_list.txt')# if split == 'train' else 'THuman/test_human_list.txt'
        with open(self.human_list) as f:
            human_dirs = f.readlines()[0:num_instance]
        # exclude_dirs = ['results_gyx_20181012_ym_2_F', 
        #                 'results_gyx_20181013_lqj_1_F', 'results_gyx_20181013_yry_1_F', 'results_gyx_20181013_znn_1_F', 'results_gyx_20181013_zsh_1_M', 
        #                 'results_gyx_20181013_zyj_1_F', 'results_gyx_20181014_sxy_2_F', 'results_gyx_20181015_dc_2_F', 'results_gyx_20181015_gh_2_F']
        # # dataset_root = '../THuman_1/nerf_data/'
        # self.root_list = [data_root] if not multi_person else [os.path.join(all_human_data_root, x.strip()) for x in human_dirs if x.strip() not in exclude_dirs]
        self.root_list = [data_root] if not multi_person else [os.path.join(all_human_data_root, x.strip()) for x in human_dirs]
        print(self.root_list)
        # print(self.root_list)
        # self.root_list = [data_root]

    def update(self, data_root):
        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']
        self.ims = np.array([
            np.array(ims_data['ims'])[self.output_view]
            for ims_data in annots['ims'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        ]) # image_name of all used images, shape (num of poses, num of output_views)
    
    def get_mask(self, pose_index, view_index):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[pose_index][view_index].replace('\\', '/').replace('jpg', 'png'))
        msk = imageio.imread(msk_path)
        msk[msk!=0]=255
        return msk

    def prepare_input_t(self, t_vertices_path):
        vertices_path = t_vertices_path
        xyz = np.load(vertices_path).astype(np.float32)

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

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate 
        params_path = os.path.join(self.data_root, "new_params", '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        R = params['R']
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        # construct point feature
        cxyz = xyz.astype(np.float32)
        # nxyz = np.zeros_like(xyz).astype(np.float32)
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

        return feature, coord, out_sh, world_bounds, bounds, vertices, params # , center, rot, trans

    def __getitem__(self, pose_index):
        """
            pose_index: [0, number of used poses), the pose index of selected poses
            view_index: [0, number of all view)
            mask_at_box_all: mask of 2D bounding box which is the 3D bounding box projection 
                training: all one array, not fixed length, no use for training
                Test: zero and one array, fixed length which can be reshape to (H,W)
            bkgd_msk_all: mask of foreground and background
                trainning: for acc loss
                test: no use
        """
        data_root_i = np.random.randint(len(self.root_list)) if self.multi_person else 0
        self.data_root = self.root_list[data_root_i]
        self.update(self.data_root)
        pose_index = pose_index % self.ni
        # print(self.data_root, pose_index)
        img_all, K_all, R_all, T_all, rgb_all, ray_o_all, ray_d_all, near_all, far_all, msk_all = [], [], [], [], [], [], [], [], [], []
        mask_at_box_all, bkgd_msk_all, img_ray_d_all = [], [], []

        # for view_index in self.output_view:
        for idx, view_index in enumerate(self.output_view):
            # Load image, mask, K, D, R, T
            img_path = os.path.join(self.data_root, self.ims[pose_index][idx].replace('\\', '/'))
            # print(img_path)
            img = np.array(imageio.imread(img_path).astype(np.float32) / 255.)
            msk = np.array(self.get_mask(pose_index, idx)) / 255.
            img[msk == 0] = 0
            K = np.array(self.cams['K'][view_index])
            D = np.array(self.cams['D'][view_index])
            img = cv2.undistort(img, K, D)
            msk = cv2.undistort(msk, K, D)
            R = np.array(self.cams['R'][view_index])
            T = np.array(self.cams['T'][view_index])
            
            # Prepare the smpl input, including the current pose and canonical pose
            if view_index == self.output_view[0]:
                # i: the pose index of all poses this person has, not the pose index of getitem input
                i = int(os.path.basename(img_path)[:-4])
                feature, coord, out_sh, world_bounds, bounds, vertices, params = self.prepare_input(i)
                if self.mean_shape:
                    path = os.path.join('m_X_template_tvertices.npy') if self.data_root[-1]=="M" else os.path.join('f_X_template_tvertices.npy')
                else:
                    path = os.path.join(self.data_root, 'X_vertices.npy')
                t_vertices = np.load(path)
                t_feature, t_coord, t_out_sh, t_bounds = self.prepare_input_t(t_vertices_path=path)
            
            # Sample rays in target space world coordinate
            rgb, ray_o, ray_d, near, far, coord_, mask_at_box, bkgd_msk, img_ray_d = if_nerf_dutils.sample_ray_THuman_batch(#sample_ray_THuman(
                    img, msk, K, R, T, world_bounds, self.nrays, self.split)

            # Pack all inputs of all views
            img = np.transpose(img, (2,0,1))
            img_ray_d = np.transpose(img_ray_d, (2,0,1))
            if view_index in self.input_view:
                img_all.append(img)
                img_ray_d_all.append(img_ray_d)
                K_all.append(K)
                R_all.append(R)
                T_all.append(T)
            msk_all.append(msk) # For test, show gt image
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
        # if self.split == 'train':
        rgb_all = np.stack(rgb_all, axis=0)
        ray_o_all = np.stack(ray_o_all, axis=0)
        ray_d_all = np.stack(ray_d_all, axis=0)
        near_all = np.stack(near_all, axis=0)[...,None]
        far_all = np.stack(far_all, axis=0)[...,None]
        mask_at_box_all = np.stack(mask_at_box_all, axis=0)
        bkgd_msk_all = np.stack(bkgd_msk_all, axis=0)
        

        ret = {
            'pose_index': pose_index, # pose_index in selected poses
            "params": params, # smpl params including smpl global R, Th
            "instance_idx": data_root_i, # person instance idx
            'gender': 1 if self.data_root[-1]=="M" else 0,

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
        return len(self.ims) * self.num_instance # * len(self.output_view)


class THumanDatasetPair(Dataset):
    def __init__(self, data_root, split='test', view_num=3, border=5, N_rand=1024*32, image_scaling=1.0, multi_person=0, num_instance=1,
                start=0, interval=10, poses_num=100, random_pair=0, male=0, mean_shape=0):
        super(THumanDatasetPair, self).__init__()
        self.data_root = data_root
        self.split = split
        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams'] # K:[(4*4), (4*4), (4*4), (4*4)], D, R, T
        self.nrays = N_rand
        self.mean_shape=mean_shape

        if view_num==8:
            self.input_view =  [0,3,6,9,12,15,18,21] # [x for x in range(24)] # [0,1,2,3]
        elif view_num==6:
            self.input_view = [0,4,8,12,16,20] # [0,1,2,3] # [x for x in range(20)]
        elif view_num==4:
            self.input_view = [0,6,12,18] # [0, 4, 12, 16]
        elif view_num==3:
            self.input_view = [4, 12, 20]
        elif view_num==12:
            self.input_view = [0,2,4,6,8,10,12,14,16,18,20,22]
         # [0,1,2,3] # [0,1, 6,7, 12,13, 18,19]
        self.train_view = [x for x in range(24)]
        self.test_view =  [x for x in range(24)] # [0,6,12,18]#
        self.output_view = self.train_view if split == 'train' else self.test_view

        self.i = start # start index 0
        self.i_intv = interval # interval 1
        self.ni = poses_num # number of used poses 30
        self.ims = np.array([
            np.array(ims_data['ims'])[self.output_view]
            for ims_data in annots['ims'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        ])

        self.multi_person = multi_person
        self.num_instance = num_instance
        all_human_data_root = os.path.join(os.path.dirname(data_root))
        self.human_list = os.path.join('data/THuman_1_male_list.txt') if male==1 else os.path.join('data/THuman_1_human_list.txt')
        # self.human_list = os.path.join('THuman_1_human_list.txt')
        # self.human_list = os.path.join(all_human_data_root, 'THuman_1_human_list.txt')# if split == 'train' else 'THuman/test_human_list.txt'
        with open(self.human_list) as f:
            human_dirs = f.readlines()[0:num_instance]
        self.root_list = [data_root] if not multi_person else [os.path.join(all_human_data_root, x.strip()) for x in human_dirs]

        self.border = border
        self.random_pair = random_pair
        self.pairs = np.loadtxt('training_set_100_pose_same_pair.txt').reshape(-1,2)
        self.img_scaling = image_scaling
        

    def update(self, data_root):
        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams'] # K:[(4*4), (4*4), (4*4), (4*4)], D, R, T
        self.ims = np.array([
            np.array(ims_data['ims'])[self.output_view]
            for ims_data in annots['ims'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        ])# image_name array, shape (150, 4)
        
    def get_mask(self, pose_index, view_index):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[pose_index][view_index].replace('\\', '/').replace('jpg', 'png'))
        msk = imageio.imread(msk_path)
        msk[msk!=0]=255
        return msk

    def prepare_input_t(self, t_vertices_path):
        vertices_path = t_vertices_path
        xyz = np.load(vertices_path).astype(np.float32)

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

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate 
        params_path = os.path.join(self.data_root, "new_params", '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        R = params['R']
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        # construct point feature
        cxyz = xyz.astype(np.float32)
        # nxyz = np.zeros_like(xyz).astype(np.float32)
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

        return feature, coord, out_sh, world_bounds, bounds, vertices, params # , center, rot, trans

    def __getitem__(self, pose_index):
        data_root_i = np.random.randint(len(self.root_list)) if self.multi_person else 0
        self.data_root = self.root_list[data_root_i]
        self.update(self.data_root)
        
        pose_index = pose_index % self.ni # poses_num

        if self.random_pair:
            # pair = [np.random.randint(0,len(self.ims)), pose_index] #source target
            thre = np.random.uniform(low=0.0, high=1.0)
            pair = [pose_index, pose_index if thre<0.5 else np.random.randint(0,len(self.ims))]
        # else:
            # pair = self.pairs[pair_index]
        sp_tp_ret = {}
        
        for k, pose_index in enumerate([int(pair[0]), int(pair[1])]):
            # load 4 views images path for pose index(index of 150 selected poses)
            img_path_list = self.ims[pose_index]
            all_img_path = []
            for i, path in enumerate(img_path_list):
                all_img_path.append(os.path.join(self.data_root, path))
            
            img_all, K_all, R_all, T_all, rgb_all, ray_o_all, ray_d_all, near_all, far_all, msk_all = [], [], [], [], [], [], [], [], [], []
            mask_at_box_all, bkgd_msk_all, img_ray_d_all = [], [], []

            for idx, view_index in enumerate(self.output_view):
                # load and resize
                img_path = os.path.join(self.data_root, self.ims[pose_index][idx].replace('\\', '/'))
                img = imageio.imread(img_path).astype(np.float32) / 255.
                msk = self.get_mask(pose_index, idx) / 255.

                # load intrinsic parameters and extrinsic paramters for each view
                img[msk == 0] = 0
                K = np.array(self.cams['K'][view_index])
                D = np.array(self.cams['D'][view_index])
                img = cv2.undistort(img, K, D)
                msk = cv2.undistort(msk, K, D)
                R = np.array(self.cams['R'][view_index])
                T = np.array(self.cams['T'][view_index])

                # load and process smpl related data
                if view_index == self.output_view[0]:
                    # i: the pose index of all poses this person has, not the pose index of getitem input
                    i = int(os.path.basename(img_path)[:-4])
                    feature, coord, out_sh, world_bounds, bounds, vertices, params = self.prepare_input(i)
                    if self.mean_shape:
                        path = os.path.join('m_X_template_tvertices.npy') if self.data_root[-1]=="M" else os.path.join('f_X_template_tvertices.npy')
                    else:
                        path = os.path.join(self.data_root, 'X_vertices.npy')
                    t_vertices = np.load(path)
                    t_feature, t_coord, t_out_sh, t_bounds = self.prepare_input_t(t_vertices_path=path)                

                # sample rays for each view
                rgb, ray_o, ray_d, near, far, coord_, mask_at_box, bkgd_msk, img_ray_d = if_nerf_dutils.sample_ray_THuman_batch(#sample_ray_THuman(
                    img, msk, K, R, T, world_bounds, self.nrays, self.split)
                                
                img = np.transpose(img, (2,0,1))
                img_ray_d = np.transpose(img_ray_d, (2,0,1))

                if view_index in self.input_view:
                    img_all.append(img)
                    img_ray_d_all.append(img_ray_d)
                    K_all.append(K)
                    R_all.append(R)
                    T_all.append(T)
                msk_all.append(msk) # For test, show gt image
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
            # if self.split == 'train':
            rgb_all = np.stack(rgb_all, axis=0)
            ray_o_all = np.stack(ray_o_all, axis=0)
            ray_d_all = np.stack(ray_d_all, axis=0)
            near_all = np.stack(near_all, axis=0)[...,None]
            far_all = np.stack(far_all, axis=0)[...,None]
            mask_at_box_all = np.stack(mask_at_box_all, axis=0)
            bkgd_msk_all = np.stack(bkgd_msk_all, axis=0)

            ret = {
                'pose_index': pose_index, # pose_index in selected poses
                "params": params, # smpl params including smpl global R, Th
                "instance_idx": data_root_i, # person instance idx
                'gender': 1 if self.data_root[-1]=="M" else 0,

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

            if k==0:
                sp_tp_ret['sp_input'] = ret
            else:
                sp_tp_ret['tp_input'] = ret
        
        return sp_tp_ret

    def __len__(self):
        # return len(self.pairs)
        return len(self.ims) * self.num_instance


class THumanDatasetBatchRandom(Dataset):
    def __init__(self, data_root, split='test', view_num=24, N_rand=1024*32, multi_person=False, num_instance=1, start=0, interval=1, poses_num=30, image_scaling=1.0, male=0, mean_shape=1):
        super(THumanDatasetBatchRandom, self).__init__()
        self.data_root = data_root
        self.split = split
        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.nrays = N_rand
        self.mean_shape = mean_shape
        if view_num==8:
            self.input_view =  [0,3,6,9,12,15,18,21] # [x for x in range(24)] # [0,1,2,3]
        elif view_num==6:
            self.input_view = [0,4,8,12,16,20] # [0,1,2,3] # [x for x in range(20)]
        elif view_num==4:
            self.input_view = [0, 4, 12, 16]
        elif view_num==3:
            self.input_view = [4, 12, 20]
        elif view_num==12:
            self.input_view = [0,2,4,6,8,10,12,14,16,18,20,22]
         # [0,1,2,3] # [0,1, 6,7, 12,13, 18,19]
        self.train_view = [x for x in range(24)]
        self.test_view =  [x for x in range(24)] # [0,6,12,18]#
        self.output_view = self.train_view if split == 'train' else self.test_view
        
        """
        annots = {
            'cams':{
                'K':[],#N arrays, (3, 3)
                'D':[],#(5, 1), all zeros
                'R':[],#(3, 3)
                'T':[] #(3, 1)
            },

            'ims':[
                # {'ims':['54138969/000000.jpg', '55011271/000000.jpg''60457274/000000.jpg']}, # same pose different views
                # {'ims':[]} 
                #  repeat time is number of poses
            ]
        }
        """
        self.i = start # start index 0
        self.i_intv = interval # interval 1
        self.ni = poses_num # number of used poses 30
        self.cams = annots['cams']
        self.ims = np.array([
            np.array(ims_data['ims'])[self.output_view]
            for ims_data in annots['ims'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        ]) # image_name of all used images, shape (num of poses, num of output_views)
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[self.output_view]
            for ims_data in annots['ims'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        ]) # view index of output_view, shape (num of poses, num of output_views)

        self.multi_person = multi_person
        self.num_instance = num_instance
        all_human_data_root = os.path.join(os.path.dirname(data_root))
        self.human_list = os.path.join('data/THuman_1_male_list.txt') if male==1 else os.path.join('data/THuman_1_human_list.txt')
        with open(self.human_list) as f:
            human_dirs = f.readlines()[0:num_instance]
        self.root_list = [data_root] if not multi_person else [os.path.join(all_human_data_root, x.strip()) for x in human_dirs]

    def update(self, data_root):
        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        self.ims = np.array([
            np.array(ims_data['ims'])[self.output_view]
            for ims_data in annots['ims'][self.i:self.i + self.ni * self.i_intv][::self.i_intv]
        ]) # image_name of all used images, shape (num of poses, num of output_views)
    
    def get_mask(self, pose_index, view_index):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[pose_index][view_index].replace('\\', '/').replace('jpg', 'png'))
        msk = imageio.imread(msk_path)
        msk[msk!=0]=255
        return msk

    def prepare_input_t(self, t_vertices_path):
        vertices_path = t_vertices_path
        xyz = np.load(vertices_path).astype(np.float32)

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

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate 
        params_path = os.path.join(self.data_root, "new_params", '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        R = params['R']
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        # construct point feature
        cxyz = xyz.astype(np.float32)
        # nxyz = np.zeros_like(xyz).astype(np.float32)
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

        return feature, coord, out_sh, world_bounds, bounds, vertices, params # , center, rot, trans

    def __getitem__(self, pose_index):
        """
            pose_index: [0, number of used poses), the pose index of selected poses
            view_index: [0, number of all view)
            mask_at_box_all: mask of 2D bounding box which is the 3D bounding box projection 
                training: all one array, not fixed length, no use for training
                Test: zero and one array, fixed length which can be reshape to (H,W)
            bkgd_msk_all: mask of foreground and background
                trainning: for acc loss
                test: no use
        """
        data_root_i = np.random.randint(len(self.root_list)) if self.multi_person else 0
        self.data_root = self.root_list[data_root_i]
        self.update(self.data_root)
        pose_index = pose_index % self.ni
        # print(self.data_root, pose_index)
        img_all, K_all, R_all, T_all, rgb_all, ray_o_all, ray_d_all, near_all, far_all, msk_all = [], [], [], [], [], [], [], [], [], []
        mask_at_box_all, bkgd_msk_all, img_ray_d_all = [], [], []

        # for view_index in self.output_view:
        for idx, view_index in enumerate(self.output_view):
            # Load image, mask, K, D, R, T
            img_path = os.path.join(self.data_root, self.ims[pose_index][idx].replace('\\', '/'))
            # print(img_path)
            img = np.array(imageio.imread(img_path).astype(np.float32) / 255.)
            msk = np.array(self.get_mask(pose_index, idx)) / 255.
            img[msk == 0] = 0
            cams = self.cams[pose_index]['cams']
            K = np.array(cams['K'][view_index])
            D = np.array(cams['D'][view_index])
            img = cv2.undistort(img, K, D)
            msk = cv2.undistort(msk, K, D)
            R = np.array(cams['R'][view_index])
            T = np.array(cams['T'][view_index])
            
            # Prepare the smpl input, including the current pose and canonical pose
            if view_index == self.output_view[0]:
                # i: the pose index of all poses this person has, not the pose index of getitem input
                i = int(os.path.basename(img_path)[:-4])
                feature, coord, out_sh, world_bounds, bounds, vertices, params = self.prepare_input(i) 
                if self.mean_shape:
                    path = os.path.join('m_X_template_tvertices.npy') if self.data_root[-1]=="M" else os.path.join('f_X_template_tvertices.npy')
                else:
                    path = os.path.join(self.data_root, 'X_vertices.npy')
                t_vertices = np.load(path)
                t_feature, t_coord, t_out_sh, t_bounds = self.prepare_input_t(t_vertices_path=path)
            
            # Sample rays in target space world coordinate
            rgb, ray_o, ray_d, near, far, coord_, mask_at_box, bkgd_msk, img_ray_d = if_nerf_dutils.sample_ray_THuman_batch(#sample_ray_THuman(
                    img, msk, K, R, T, world_bounds, self.nrays, self.split)

            # Pack all inputs of all views
            img = np.transpose(img, (2,0,1))
            img_ray_d = np.transpose(img_ray_d, (2,0,1))
            if view_index in self.input_view:
                img_all.append(img)
                img_ray_d_all.append(img_ray_d)
                K_all.append(K)
                R_all.append(R)
                T_all.append(T)
            msk_all.append(msk) # For test, show gt image
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
        # if self.split == 'train':
        rgb_all = np.stack(rgb_all, axis=0)
        ray_o_all = np.stack(ray_o_all, axis=0)
        ray_d_all = np.stack(ray_d_all, axis=0)
        near_all = np.stack(near_all, axis=0)[...,None]
        far_all = np.stack(far_all, axis=0)[...,None]
        mask_at_box_all = np.stack(mask_at_box_all, axis=0)
        bkgd_msk_all = np.stack(bkgd_msk_all, axis=0)
        

        ret = {
            'pose_index': pose_index, # pose_index in selected poses
            "params": params, # smpl params including smpl global R, Th
            "instance_idx": data_root_i, # person instance idx
            'gender': 1 if self.data_root[-1]=="M" else 0,

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
        return len(self.ims) * self.num_instance # * len(self.output_view)

