# -*- coding: utf-8 -*
import os
import numpy as np
import imageio
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, dataset
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.run_nerf_helpers import *
from parser_config import *
from lib.h36m_dataset import H36MDataset, H36MDatasetBatch, H36MDatasetPair, H36MDatasetBatchAll
from lib.THuman_dataset import THumanDataset, THumanDatasetBatch, THumanDatasetPair
import torch.distributions as tdist
from model_selection import *
from skimage.measure import compare_ssim

import json 

parser = config_parser()
global_args = parser.parse_args()


def psnr_metric(img_pred, img_gt):
    mse = np.mean((img_pred - img_gt)**2)
    psnr = -10 * np.log(mse) / np.log(10)
    return psnr


def ssim_metric(rgb_pred, rgb_gt, mask_at_box, H, W):
    # convert the pixels into an image
    img_pred = np.zeros((H, W, 3))
    img_pred[mask_at_box] = rgb_pred
    img_gt = np.zeros((H, W, 3))
    img_gt[mask_at_box] = rgb_gt

    # crop the object region
    x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
    img_pred = img_pred[y:y + h, x:x + w]
    img_gt = img_gt[y:y + h, x:x + w]
    
    # compute the ssim
    ssim = compare_ssim(img_pred, img_gt, multichannel=True)
    
    return ssim


def test_THuman_ssim(chunk, render_kwargs, savedir=None, global_args=None, device=None, render=None):

    batch_size = 1 # global_args.batch_size
    interval = 1 #1
    novel_view = [1,5,7,11,13,17,19,23] # [x for x in range(24)] # [1,5,7,11,13,17,19,23]
    poses_num = 5 #5 # batch_size
    data_root_list = [
     "./data/THuman/nerf_data_/results_gyx_20181012_sty_1_M",
     "./data/THuman/nerf_data_/results_gyx_20181012_xsx_2_M",
     "./data/THuman/nerf_data_/results_gyx_20181013_hyd_1_M",
     "./data/THuman/nerf_data_/results_gyx_20181012_lw_2_F",
     "./data/THuman/nerf_data_/results_gyx_20181013_xyz_1_F",
    ]
    start_pose_list = [20, 24, 27, 25, 28]
    start_person = 25
    end_person = 30 #30
    H, W = 512, 512
    
    # wo finetune 10 person 10 pose 24 view --> 2400 images
    #  w finetune 10 person 10 pose 24 view --> 2400 images
    # human_list = os.path.join('THuman_1_male_list.txt') 
    all_human_data_root = os.path.join(os.path.dirname(global_args.data_root))
    human_list = os.path.join('./data/THuman_1_human_list.txt')
    with open(human_list) as f:
        test_list = f.readlines()[start_person:end_person]
    test_THuman_list = [os.path.join(all_human_data_root, x.strip()) for x in test_list]

    metric = {
            "novel_view_mean_human":[], "novel_view_all_human":[], "novel_view_mse":[], "novel_view_psnr":[], "novel_view_ssim":[], 
            "novel_pose_mean_human":[], "novel_pose_all_human":[], "novel_pose_mse":[], "novel_pose_psnr":[], "novel_pose_ssim":[],
            "all_human_names":[]
            }
    
    all_human_psnr = []
    all_human_mse = []
    all_human_ssim = []
    all_human_name = []
    for p, human_data_path in enumerate(test_THuman_list):
        all_human_name.append(test_list[p].strip())
        data_root = human_data_path
        start_pose = start_pose_list[p]

        # novel pose novel view test
        test_set = THumanDatasetBatch(data_root, split=global_args.test_split, view_num=global_args.view_num,\
            start=start_pose, interval=1, poses_num=poses_num+1, model=global_args.model, male=global_args.male, mean_shape=global_args.mean_shape) # hyd 13 lc 5
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False) # global_args.num_worker
        sp_input = None
        
        human_save_path = os.path.join(savedir, "novel_pose", test_list[p].strip())
        os.makedirs(human_save_path, exist_ok=True)
        all_pose_psnr = []
        all_pose_mse = []
        all_pose_ssim = []
        for i, data in enumerate(test_loader):
            data = to_cuda(device, data)
            if i==0:
                sp_input = data
                continue
            tp_input = data
            all_view_psnr = []
            all_view_mse = []
            all_view_ssim = []
            for k in range(0, tp_input['rgb_all'].shape[1], interval):
                if k not in novel_view:
                    continue
                batch_rays = torch.stack([tp_input['ray_o_all'][:,k], tp_input['ray_d_all'][:,k]], 1) # (B,2,N_rand,3)
                near=tp_input['near_all'][:,k]
                far=tp_input['far_all'][:,k]
                target_s = tp_input['rgb_all'][:, k]
                msk = tp_input['msk_all'][:,k]
                mask_at_box = tp_input['mask_at_box_all'][:,k]

                # use mask_at_box to discard unimportant pixel

                time_0 = time.time()
                rgb, disp, acc, extras = render(chunk=chunk, rays=batch_rays, sp_input=sp_input, tp_input=tp_input,
                                                near=near, far=far, **render_kwargs)
                time_1 = time.time()
                print("Time per image: ", time_1 - time_0)

                rgb = rgb.reshape(batch_size,512,512,3)
                target_s = target_s.reshape(batch_size,512,512,3)
                msk = msk.reshape(batch_size,512,512)
                mask_at_box = mask_at_box.reshape(batch_size,512,512)
                
                # if savedir is not None:
                for j in range(batch_size):
                    img_pred = rgb[j]
                    gt_img = target_s[j]
                    pred_rgb8 = to8b(img_pred.cpu().numpy())
                    gt_rgb8 = to8b(gt_img.cpu().numpy())
                    # rgb8 = np.concatenate([pred_rgb8, gt_rgb8], axis=1)
                    # gt_filename = os.path.join(human_save_path, '{:03d}_{:03d}_{:03d}_gt.png'.format(int(sp_input['pose_index'][j]), int(tp_input['pose_index'][j]), k))
                    # pred_filename = os.path.join(human_save_path, '{:03d}_{:03d}_{:03d}_pred.png'.format(int(sp_input['pose_index'][j]), int(tp_input['pose_index'][j]), k))
                    gt_filename = os.path.join(human_save_path, 'frame{:04d}_view{:04d}_gt.png'.format(int(tp_input['pose_index'][j])+start_pose, k))
                    pred_filename = os.path.join(human_save_path, 'frame{:04d}_view{:04d}.png'.format(int(tp_input['pose_index'][j])+start_pose, k))
                    imageio.imwrite(gt_filename, gt_rgb8)
                    imageio.imwrite(pred_filename, pred_rgb8)
                    # filename = os.path.join(savedir, '{:02d}_{:02d}_{:02d}.png'.format(int(sp_input['pose_index'][j]), int(tp_input['pose_index'][j]), k))
                    # imageio.imwrite(filename, rgb8)
                
                    mse = img2mse(img_pred[mask_at_box[j]], gt_img[mask_at_box[j]])
                    # psnr = round(mse2psnr(mse).item(), 5)
                    psnr = psnr_metric(img_pred[mask_at_box[j]].cpu().numpy(), gt_img[mask_at_box[j]].cpu().numpy())
                    ssim = ssim_metric(img_pred[mask_at_box[j]].cpu().numpy(), gt_img[mask_at_box[j]].cpu().numpy(), mask_at_box[j].cpu().numpy(), H, W)
                    # ssim = compare_ssim(pred_rgb8, gt_rgb8, multichannel=True)
                    print("[Test] ", "human: ", p, " pose:", int(sp_input['pose_index'][j]), int(tp_input['pose_index'][j]), " view:", k, \
                        " mse:", round(mse.item(), 5), " psnr:", {psnr}, " ssim:", {ssim})
                    all_view_mse.append(mse.item())
                    all_view_psnr.append(psnr)
                    all_view_ssim.append(ssim)

                    input_img = sp_input['img_all'][j].cpu().numpy().transpose(2,0,3,1).reshape(512, -1, 3) * 255. #NCHW->HNWC
                    # input_img = cv2.resize(input_img, (512*2, int(input_img.shape[0] * 512*2 / input_img.shape[1])))
                    # img = image_add_text(filename, 'PSNR: %f' % (psnr), 20, 20, text_color=(255, 255, 255), text_size=20)
                    # img = np.concatenate([input_img, img], axis=0)
                    # filename = os.path.join(human_save_path, "input_images.png")
                    filename = os.path.join(human_save_path, "input_images_{:02d}.png".format(i))
                    imageio.imwrite(filename, to8b(input_img/255.))

            all_pose_mse.append(all_view_mse)
            all_pose_psnr.append(all_view_psnr)
            all_pose_ssim.append(all_view_ssim)

        all_human_psnr.append(all_pose_psnr)
        all_human_mse.append(all_pose_mse)
        all_human_ssim.append(all_pose_ssim) # human * pose * novel_view (5,5,8)

    metric["all_human_names"] = all_human_name
    human_num = len(all_human_psnr)
    metric["novel_pose_mse"] = np.array(all_human_mse)
    metric["novel_pose_psnr"] = np.array(all_human_psnr)
    metric["novel_pose_ssim"] = np.array(all_human_ssim)
    metric["novel_pose_mean_human"] = np.array([np.mean(metric["novel_pose_mse"][:, :, :]), np.mean(metric["novel_pose_psnr"][:, :, :]), np.mean(metric["novel_pose_ssim"][:, :, :])])
    metric["novel_pose_all_human"] = np.array([
        np.mean(metric["novel_pose_mse"][:, :, :].reshape(human_num, -1), axis=-1), 
        np.mean(metric["novel_pose_psnr"][:, :, :].reshape(human_num, -1), axis=-1), 
        np.mean(metric["novel_pose_ssim"][:, :, :].reshape(human_num, -1), axis=-1)
        ])
    # metric["mean_novel_view"] = np.array([np.mean(metric["mse"][:, 0, :]), np.mean(metric["psnr"][:, 0, :]), np.mean(metric["ssim"][:, 0, :])])
    # metric["mean_novel_pose"] = np.array([np.mean(metric["mse"][:, 1:, :]), np.mean(metric["psnr"][:, 1:, :]), np.mean(metric["ssim"][:, 1:, :])])

    all_human_psnr = []
    all_human_mse = []
    all_human_ssim = []
    for p, human_data_path in enumerate(test_THuman_list):
        data_root = human_data_path
        start_pose = 0

        # novel pose novel view test
        test_set = THumanDatasetBatch(data_root, split=global_args.test_split, view_num=global_args.view_num,\
            start=start_pose, interval=1, poses_num=poses_num,model=global_args.model, male=global_args.male, mean_shape=global_args.mean_shape) # hyd 13 lc 5
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False) # global_args.num_worker
        
        human_save_path = os.path.join(savedir, "novel_view", test_list[p].strip())
        os.makedirs(human_save_path, exist_ok=True)
        all_pose_psnr = []
        all_pose_mse = []
        all_pose_ssim = []
        for i, data in enumerate(test_loader):
            data = to_cuda(device, data)
            sp_input = data
            tp_input = data
            all_view_psnr = []
            all_view_mse = []
            all_view_ssim = []
            for k in range(0, tp_input['rgb_all'].shape[1], interval):
                if k not in novel_view:
                    continue
                batch_rays = torch.stack([tp_input['ray_o_all'][:,k], tp_input['ray_d_all'][:,k]], 1) # (B,2,N_rand,3)
                near=tp_input['near_all'][:,k]
                far=tp_input['far_all'][:,k]
                target_s = tp_input['rgb_all'][:, k]
                msk = tp_input['msk_all'][:,k]
                mask_at_box = tp_input['mask_at_box_all'][:,k]

                # use mask_at_box to discard unimportant pixel

                time_0 = time.time()
                rgb, disp, acc, extras = render(chunk=chunk, rays=batch_rays, sp_input=sp_input, tp_input=tp_input,
                                                near=near, far=far, **render_kwargs)
                time_1 = time.time()
                print("Time per image: ", time_1 - time_0)

                rgb = rgb.reshape(batch_size,512,512,3)
                target_s = target_s.reshape(batch_size,512,512,3)
                msk = msk.reshape(batch_size,512,512)
                mask_at_box = mask_at_box.reshape(batch_size,512,512)
                
                # if savedir is not None:
                for j in range(batch_size):
                    img_pred = rgb[j]
                    gt_img = target_s[j]
                    pred_rgb8 = to8b(img_pred.cpu().numpy())
                    gt_rgb8 = to8b(gt_img.cpu().numpy())
                    # rgb8 = np.concatenate([pred_rgb8, gt_rgb8], axis=1)
                    # gt_filename = os.path.join(human_save_path, '{:03d}_{:03d}_{:03d}_gt.png'.format(int(sp_input['pose_index'][j]), int(tp_input['pose_index'][j]), k))
                    # pred_filename = os.path.join(human_save_path, '{:03d}_{:03d}_{:03d}_pred.png'.format(int(sp_input['pose_index'][j]), int(tp_input['pose_index'][j]), k))
                    gt_filename = os.path.join(human_save_path, 'frame{:04d}_view{:04d}_gt.png'.format(int(tp_input['pose_index'][j]), k))
                    pred_filename = os.path.join(human_save_path, 'frame{:04d}_view{:04d}.png'.format(int(tp_input['pose_index'][j]), k))
                    imageio.imwrite(gt_filename, gt_rgb8)
                    imageio.imwrite(pred_filename, pred_rgb8)
                    # filename = os.path.join(savedir, '{:02d}_{:02d}_{:02d}.png'.format(int(sp_input['pose_index'][j]), int(tp_input['pose_index'][j]), k))
                    # imageio.imwrite(filename, rgb8)
                
                    mse = img2mse(img_pred[mask_at_box[j]], gt_img[mask_at_box[j]])
                    psnr = psnr_metric(img_pred[mask_at_box[j]].cpu().numpy(), gt_img[mask_at_box[j]].cpu().numpy())
                    ssim = ssim_metric(img_pred[mask_at_box[j]].cpu().numpy(), gt_img[mask_at_box[j]].cpu().numpy(), mask_at_box[j].cpu().numpy(), H, W)
                    # psnr = round(mse2psnr(mse).item(), 5)
                    # ssim = compare_ssim(pred_rgb8, gt_rgb8, multichannel=True)
                    print("[Test] ", "human: ", p, " pose:", int(sp_input['pose_index'][j]), int(tp_input['pose_index'][j]), " view:", k, \
                        " mse:", round(mse.item(), 5), " psnr:", {psnr}, " ssim:", {ssim})
                    all_view_mse.append(mse.item())
                    all_view_psnr.append(psnr)
                    all_view_ssim.append(ssim)

                    input_img = sp_input['img_all'][j].cpu().numpy().transpose(2,0,3,1).reshape(512, -1, 3) * 255. #NCHW->HNWC
                    # input_img = cv2.resize(input_img, (512*2, int(input_img.shape[0] * 512*2 / input_img.shape[1])))
                    # img = image_add_text(filename, 'PSNR: %f' % (psnr), 20, 20, text_color=(255, 255, 255), text_size=20)
                    # img = np.concatenate([input_img, img], axis=0)
                    filename = os.path.join(human_save_path, "input_images_{:02d}.png".format(i))
                    imageio.imwrite(filename, to8b(input_img/255.))

            all_pose_mse.append(all_view_mse)
            all_pose_psnr.append(all_view_psnr)
            all_pose_ssim.append(all_view_ssim)

        all_human_psnr.append(all_pose_psnr)
        all_human_mse.append(all_pose_mse)
        all_human_ssim.append(all_pose_ssim) # human * pose * novel_view (5,5,8)

    human_num = len(all_human_psnr)
    metric["novel_view_mse"] = np.array(all_human_mse)
    metric["novel_view_psnr"] = np.array(all_human_psnr)
    metric["novel_view_ssim"] = np.array(all_human_ssim)
    metric["novel_view_mean_human"] = np.array([np.mean(metric["novel_view_mse"][:, :, :]), np.mean(metric["novel_view_psnr"][:, :, :]), np.mean(metric["novel_view_ssim"][:, :, :])])
    metric["novel_view_all_human"] = np.array([
        np.mean(metric["novel_view_mse"][:, :, :].reshape(human_num, -1), axis=-1), 
        np.mean(metric["novel_view_psnr"][:, :, :].reshape(human_num, -1), axis=-1), 
        np.mean(metric["novel_view_ssim"][:, :, :].reshape(human_num, -1), axis=-1)
        ])

    metric_json = {}
    with open(savedir+"/metrics.json", 'w') as f:
        metric_json["novel_view_mean_human"] = metric["novel_view_mean_human"].tolist()
        metric_json["novel_pose_mean_human"] = metric["novel_pose_mean_human"].tolist()
        metric_json["novel_view_all_human"] = metric["novel_view_all_human"].tolist()
        metric_json["novel_pose_all_human"] = metric["novel_pose_all_human"].tolist()
        
        json.dump(metric_json, f)

    np.save(savedir+"/metrics.npy", metric)

    return


def test_H36M(chunk, render_kwargs, savedir=None, global_args=None, device=None, render=None, test_persons=2):

    batch_size = 1 # global_args.batch_size
    interval = 1 #1
    novel_view = [3] # [x for x in range(24)] # [1,5,7,11,13,17,19,23]
    
    all_poses_num_list = [49, 127, 83, 200, 87, 133, 82]
    all_novel_view_poses_num_list = [150, 250, 150, 300, 250, 260, 200]
    all_input_pose_list = [250, 30, 1050, 820, 370, 20, 20]
    all_start_pose_list = [750, 1250, 750, 1500, 1250, 1300, 1000]
    all_test_list = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]
    all_test_H36M_list = ["./data/h36m/S1/Posing", "./data/h36m/S5/Posing", "./data/h36m/S6/Posing", "./data/h36m/S7/Posing", 
                        "./data/h36m/S8/Posing", "./data/h36m/S9/Posing", "./data/h36m/S11/Posing"]

    poses_num_list = [all_poses_num_list[int(test_persons)]]
    novel_view_poses_num_list = [all_novel_view_poses_num_list[int(test_persons)]]
    input_pose_list = [all_input_pose_list[int(test_persons)]]
    start_pose_list = [all_start_pose_list[int(test_persons)]]
    test_list = [all_test_list[int(test_persons)]]
    test_H36M_list = [all_test_H36M_list[int(test_persons)]]

    metric = {
            "novel_view_mean_human":[], "novel_view_all_human":[], "novel_view_mse":[], "novel_view_psnr":[], "novel_view_ssim":[], 
            "novel_pose_mean_human":[], "novel_pose_all_human":[], "novel_pose_mse":[], "novel_pose_psnr":[], "novel_pose_ssim":[],
            "all_human_names":[]
            }
    
    all_human_psnr = []
    all_human_mse = []
    all_human_ssim = []
    all_human_name = []
    for p, human_data_path in enumerate(test_H36M_list):
        all_human_name.append(test_list[p].strip())
        data_root = human_data_path
        input_pose = input_pose_list[p]
        start_pose = start_pose_list[p]
        poses_num = poses_num_list[p]
        sp_input = None

        ### input pose sp_input 
        sp_test_set = H36MDatasetBatch(data_root, split=global_args.test_split, view_num=global_args.view_num,\
            start=input_pose, interval=1, poses_num=1, mean_shape=global_args.mean_shape, new_mask=global_args.new_mask)
        sp_test_loader = DataLoader(sp_test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=False) # global_args.num_worker
        for _, data in enumerate(sp_test_loader):
            sp_input = to_cuda(device, data)

        ### novel pose novel view test
        test_set = H36MDatasetBatch(data_root, split=global_args.test_split, view_num=global_args.view_num,\
            start=start_pose, interval=5, poses_num=poses_num, mean_shape=global_args.mean_shape, new_mask=global_args.new_mask)
        H,W = 1000, 1000
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False) # global_args.num_worker
        
        human_save_path = os.path.join(savedir, "novel_pose", test_list[p].strip())
        os.makedirs(human_save_path, exist_ok=True)
        all_pose_psnr = []
        all_pose_mse = []
        all_pose_ssim = []
        for i, data in enumerate(test_loader):
            data = to_cuda(device, data)
            tp_input = data
            all_view_psnr = []
            all_view_mse = []
            all_view_ssim = []
            for k in range(0, tp_input['rgb_all'].shape[1], interval):
                if k not in novel_view:
                    continue
                batch_rays = torch.stack([tp_input['ray_o_all'][:,k], tp_input['ray_d_all'][:,k]], 1) # (B,2,N_rand,3)
                near=tp_input['near_all'][:,k]
                far=tp_input['far_all'][:,k]
                target_s = tp_input['rgb_all'][:, k]
                msk = tp_input['msk_all'][:,k]
                mask_at_box = tp_input['mask_at_box_all'][:,k]

                # use mask_at_box to discard unimportant pixel
                near = near[mask_at_box].unsqueeze(0)
                far = far[mask_at_box].unsqueeze(0)
                target_s = target_s[mask_at_box].unsqueeze(0)
                batch_rays = torch.stack([tp_input['ray_o_all'][:,k][mask_at_box], tp_input['ray_d_all'][:,k][mask_at_box]], 0).unsqueeze(0)

                # render_kwargs['network_fn'].module.training False
                time_0 = time.time()
                rgb, disp, acc, extras = render(chunk=chunk, rays=batch_rays, sp_input=sp_input, tp_input=tp_input,
                                                near=near, far=far, **render_kwargs)
                time_1 = time.time()
                # print("Time per image: ", time_1 - time_0)

                ### single GPU
                mask_at_box = mask_at_box.reshape(batch_size,H,W).detach().cpu().numpy()
                img_pred_all = np.zeros((batch_size, H, W, 3))
                img_pred_all[mask_at_box] = rgb.detach().cpu().numpy()
                img_gt_all = np.zeros((batch_size, H, W, 3))
                # img_gt_all[mask_at_box] = target_s.detach().cpu().numpy()
                img_gt_all = tp_input['o_img_all'][:, k].cpu().numpy().transpose(0,2,3,1)
                msk = msk.reshape(batch_size,H,W)

                # rgb = rgb.reshape(batch_size,H,W,3)
                # target_s = target_s.reshape(batch_size,H,W,3)
                # msk = msk.reshape(batch_size,H,W)
                # mask_at_box = mask_at_box.reshape(batch_size,H,W)
                
                # if savedir is not None:
                for j in range(batch_size):
                    # img_pred = rgb[j]
                    # gt_img = target_s[j]
                    img_pred = img_pred_all[j]
                    gt_img = img_gt_all[j]
                    pred_rgb8 = to8b(img_pred)
                    gt_rgb8 = to8b(gt_img)
                    gt_filename = os.path.join(human_save_path, 'frame{:04d}_view{:04d}_gt.png'.format(int(tp_input['pose_index'][j])*5+start_pose, k))
                    pred_filename = os.path.join(human_save_path, 'frame{:04d}_view{:04d}.png'.format(int(tp_input['pose_index'][j])*5+start_pose, k))
                    imageio.imwrite(gt_filename, gt_rgb8)
                    imageio.imwrite(pred_filename, pred_rgb8)
                    # filename = os.path.join(savedir, '{:02d}_{:02d}_{:02d}.png'.format(int(sp_input['pose_index'][j]), int(tp_input['pose_index'][j]), k))
                    # imageio.imwrite(filename, rgb8)
                
                    # mse = img2mse(img_pred[mask_at_box[j]], gt_img[mask_at_box[j]])
                    mse = img2mse(torch.tensor(img_pred[mask_at_box[j]]), torch.tensor(gt_img[mask_at_box[j]]))
                    # psnr = round(mse2psnr(mse).item(), 5)
                    psnr = psnr_metric(img_pred[mask_at_box[j]], gt_img[mask_at_box[j]])
                    # ssim = compare_ssim(pred_rgb8, gt_rgb8, multichannel=True)
                    # ssim = ssim_metric(img_pred, gt_img, mask_at_box[j], H, W)
                    ssim = ssim_metric(img_pred[mask_at_box[j]], gt_img[mask_at_box[j]], mask_at_box[j], H, W)
                    print("[Test] ", "human: ", p, " pose:", int(sp_input['pose_index'][j]), int(tp_input['pose_index'][j]), " view:", k, \
                        " mse:", round(mse.item(), 5), " psnr:", {psnr}, " ssim:", {ssim})
                    all_view_mse.append(mse.item())
                    all_view_psnr.append(psnr)
                    all_view_ssim.append(ssim)

                    input_img = sp_input['img_all'][j].cpu().numpy().transpose(2,0,3,1).reshape(H, -1, 3) * 255. #NCHW->HNWC
                    filename = os.path.join(human_save_path, "input_images_{:04d}.png".format(input_pose))
                    imageio.imwrite(filename, to8b(input_img/255.))

            all_pose_mse.append(all_view_mse)
            all_pose_psnr.append(all_view_psnr)
            all_pose_ssim.append(all_view_ssim)

        all_human_psnr.append(all_pose_psnr)
        all_human_mse.append(all_pose_mse)
        all_human_ssim.append(all_pose_ssim) # human * pose * novel_view (5,5,8)

    metric["all_human_names"] = all_human_name
    human_num = len(all_human_psnr)
    metric["novel_pose_mse"] = np.array(all_human_mse, dtype=object)
    metric["novel_pose_psnr"] = np.array(all_human_psnr, dtype=object)
    metric["novel_pose_ssim"] = np.array(all_human_ssim, dtype=object)
    
    metric["novel_pose_all_human"] = np.array([
        [np.mean(metric["novel_pose_mse"][0])], 
        [np.mean(metric["novel_pose_psnr"][0])], 
        [np.mean(metric["novel_pose_ssim"][0])]
        ])
    
    ### novel view test ###
    all_human_psnr = []
    all_human_mse = []
    all_human_ssim = []
    for p, human_data_path in enumerate(test_H36M_list):
        data_root = human_data_path
        start_pose = 0
        poses_num = novel_view_poses_num_list[p]

        ### novel pose novel view test
        test_set = H36MDatasetBatch(data_root, split=global_args.test_split, view_num=global_args.view_num,\
            start=start_pose, interval=5, poses_num=poses_num, mean_shape=global_args.mean_shape, new_mask=global_args.new_mask)
        H,W = 1000, 1000
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False) # global_args.num_worker
        
        human_save_path = os.path.join(savedir, "novel_view", test_list[p].strip())
        os.makedirs(human_save_path, exist_ok=True)
        all_pose_psnr = []
        all_pose_mse = []
        all_pose_ssim = []
        for i, data in enumerate(test_loader):
            data = to_cuda(device, data)
            sp_input = data
            tp_input = data
            all_view_psnr = []
            all_view_mse = []
            all_view_ssim = []
            for k in range(0, tp_input['rgb_all'].shape[1], interval):
                if k not in novel_view:
                    continue
                batch_rays = torch.stack([tp_input['ray_o_all'][:,k], tp_input['ray_d_all'][:,k]], 1) # (B,2,N_rand,3)
                near=tp_input['near_all'][:,k]
                far=tp_input['far_all'][:,k]
                target_s = tp_input['rgb_all'][:, k]
                msk = tp_input['msk_all'][:,k]
                mask_at_box = tp_input['mask_at_box_all'][:,k]

                # use mask_at_box to discard unimportant pixel
                near = near[mask_at_box].unsqueeze(0)
                far = far[mask_at_box].unsqueeze(0)
                target_s = target_s[mask_at_box].unsqueeze(0)
                batch_rays = torch.stack([tp_input['ray_o_all'][:,k][mask_at_box], tp_input['ray_d_all'][:,k][mask_at_box]], 0).unsqueeze(0)

                # render_kwargs['network_fn'].module.training False
                rgb, disp, acc, extras = render(chunk=chunk, rays=batch_rays, sp_input=sp_input, tp_input=tp_input,
                                                near=near, far=far, **render_kwargs)

                ### single GPU
                mask_at_box = mask_at_box.reshape(batch_size,H,W).detach().cpu().numpy()
                img_pred_all = np.zeros((batch_size, H, W, 3))
                img_pred_all[mask_at_box] = rgb.detach().cpu().numpy()
                img_gt_all = np.zeros((batch_size, H, W, 3))
                # img_gt_all[mask_at_box] = target_s.detach().cpu().numpy()
                img_gt_all = tp_input['o_img_all'][:, k].cpu().numpy().transpose(0,2,3,1)
                msk = msk.reshape(batch_size,H,W)

                # if savedir is not None:
                for j in range(batch_size):
                    img_pred = img_pred_all[j]
                    gt_img = img_gt_all[j]
                    pred_rgb8 = to8b(img_pred)
                    gt_rgb8 = to8b(gt_img)
                    # gt_filename = os.path.join(human_save_path, '{:03d}_{:03d}_{:03d}_gt.png'.format(int(sp_input['pose_index'][j]), int(tp_input['pose_index'][j]), k))
                    # pred_filename = os.path.join(human_save_path, '{:03d}_{:03d}_{:03d}_pred.png'.format(int(sp_input['pose_index'][j]), int(tp_input['pose_index'][j]), k))
                    gt_filename = os.path.join(human_save_path, 'frame{:04d}_view{:04d}_gt.png'.format(int(tp_input['pose_index'][j])*5, k))
                    pred_filename = os.path.join(human_save_path, 'frame{:04d}_view{:04d}.png'.format(int(tp_input['pose_index'][j])*5, k))
                    imageio.imwrite(gt_filename, gt_rgb8)
                    imageio.imwrite(pred_filename, pred_rgb8)
                    # filename = os.path.join(savedir, '{:02d}_{:02d}_{:02d}.png'.format(int(sp_input['pose_index'][j]), int(tp_input['pose_index'][j]), k))
                    # imageio.imwrite(filename, rgb8)
                
                    # mse = img2mse(img_pred[mask_at_box[j]], gt_img[mask_at_box[j]])
                    mse = img2mse(torch.tensor(img_pred[mask_at_box[j]]), torch.tensor(gt_img[mask_at_box[j]]))
                    # psnr = round(mse2psnr(mse).item(), 5)
                    psnr = psnr_metric(img_pred[mask_at_box[j]], gt_img[mask_at_box[j]])
                    # ssim = compare_ssim(pred_rgb8, gt_rgb8, multichannel=True)
                    # ssim = ssim_metric(img_pred, gt_img, mask_at_box[j], H, W)
                    ssim = ssim_metric(img_pred[mask_at_box[j]], gt_img[mask_at_box[j]], mask_at_box[j], H, W)
                    print("[Test] ", "human: ", p, " pose:", int(sp_input['pose_index'][j]), int(tp_input['pose_index'][j]), " view:", k, \
                        " mse:", round(mse.item(), 5), " psnr:", {psnr}, " ssim:", {ssim})
                    all_view_mse.append(mse.item())
                    all_view_psnr.append(psnr)
                    all_view_ssim.append(ssim)

                    input_img = sp_input['img_all'][j].cpu().numpy().transpose(2,0,3,1).reshape(H, -1, 3) * 255. #NCHW->HNWC
                    # input_img = cv2.resize(input_img, (512*2, int(input_img.shape[0] * 512*2 / input_img.shape[1])))
                    # img = image_add_text(filename, 'PSNR: %f' % (psnr), 20, 20, text_color=(255, 255, 255), text_size=20)
                    # img = np.concatenate([input_img, img], axis=0)
                    # filename = os.path.join(human_save_path, "input_images_{:02d}.png".format(i))
                    # imageio.imwrite(filename, to8b(input_img/255.))

            all_pose_mse.append(all_view_mse)
            all_pose_psnr.append(all_view_psnr)
            all_pose_ssim.append(all_view_ssim)

        all_human_psnr.append(all_pose_psnr)
        all_human_mse.append(all_pose_mse)
        all_human_ssim.append(all_pose_ssim) # human * pose * novel_view (5,5,8)

    human_num = len(all_human_psnr)
    metric["novel_view_mse"] = np.array(all_human_mse, dtype=object)
    metric["novel_view_psnr"] = np.array(all_human_psnr, dtype=object)
    metric["novel_view_ssim"] = np.array(all_human_ssim, dtype=object)
    
    metric["novel_view_all_human"] = np.array([
        [np.mean(metric["novel_view_mse"][0])], 
        [np.mean(metric["novel_view_psnr"][0])], 
        [np.mean(metric["novel_view_ssim"][0])]
        ])
    metric_json = {}
    with open(savedir+"/metrics.json", 'w') as f:
        # metric_json["novel_view_mean_human"] = metric["novel_view_mean_human"].tolist()
        # metric_json["novel_pose_mean_human"] = metric["novel_pose_mean_human"].tolist()
        metric_json["novel_view_all_human"] = metric["novel_view_all_human"].tolist()
        metric_json["novel_pose_all_human"] = metric["novel_pose_all_human"].tolist()
        metric_json["all_human_names"] = metric["all_human_names"]
        
        json.dump(metric_json, f)

    np.save(savedir+"/metrics.npy", metric)
    
    return

