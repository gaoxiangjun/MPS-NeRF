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
from lib.all_test import test_H36M, test_THuman_ssim

parser = config_parser()
global_args = parser.parse_args()

if global_args.ddp:
    torch.cuda.set_device(global_args.local_rank)
    dist.init_process_group(backend='nccl')
    
if global_args.use_os_env:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_args.batch_size = torch.cuda.device_count()

perturb_distri = tdist.Normal(torch.tensor([0., 0., 0.]), torch.tensor([0.01]))
density_actfn = shifted_softplus
rgb_actfn = wide_sigmoid

setup_seed(0)

def run_network(inputs, viewdirs, fn, sp_input=None, tp_input=None):#, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [inputs.shape[0], -1, inputs.shape[-1]])

    if viewdirs is not None:
        input_dirs = viewdirs[:,:,None].expand(inputs.shape) # torch.Size([1024, 64, 3])
        input_dirs_flat = torch.reshape(input_dirs, [inputs.shape[0], -1, input_dirs.shape[-1]]) # torch.Size([65536, 3])

    outputs_flat = fn(sp_input, tp_input, inputs_flat, input_dirs_flat)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])

    normal_smooth_loss = 0.0
    correction_smooth_loss = 0.0
    smpl_normal_loss = 0.0
    other_loss = torch.zeros(1,4).cuda()#.reshape(1,1)
    sh = inputs_flat.shape

    if fn.module.training and global_args.smooth_loss:
        intv_flag = (int(sp_input['global_step'][0].item()) % int(sp_input['smooth_interval'][0].item()))==0
        if intv_flag:
            # TODO pts mask
            delta_x = perturb_distri.sample(sh[:-1]).cuda()#.requires_grad_()
            inputs_flat_1 = inputs_flat + delta_x

            outputs_flat_1 = fn(sp_input, tp_input, inputs_flat_1, input_dirs_flat)
            normal_smooth_loss += img2mse(outputs_flat_1[..., 17:20], outputs_flat[..., 17:20])
            smpl_normal_loss += img2mse(outputs_flat[..., 20:23], -outputs_flat[..., 17:20])
            
            # correction_smooth_loss += img2mse(outputs_flat_1[..., 5:8], outputs_flat[..., 5:8]) + img2mse(outputs_flat_1[..., 8:11], outputs_flat[..., 8:11])        
            # normal_smooth_loss += img2mse(outputs_flat_1[..., 14:17], outputs_flat[..., 14:17])
            # smpl_normal_loss += img2mse(outputs_flat[..., 17:20], -outputs_flat[..., 14:17])

            other_loss[0][0] += 0.1 * normal_smooth_loss + correction_smooth_loss * 10 + 0.1 * smpl_normal_loss
            other_loss[0][1] += normal_smooth_loss # + 0.01 * sparsity_loss
            other_loss[0][2] += correction_smooth_loss # + 0.01 * sparsity_loss
            other_loss[0][3] += smpl_normal_loss
            other_loss = other_loss.reshape(1,4)

    # others = torch.reshape(others, list(inputs.shape[:-1]) + [others.shape[-1]])
    return outputs, other_loss


def batchify_rays(rays_flat, chunk=1024*32, sp_input=None, tp_input=None, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[1], chunk):
        ret = render_rays(rays_flat[:, i:i+chunk], sp_input=sp_input, tp_input=tp_input, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
        torch.cuda.empty_cache()
    all_ret = {k : torch.cat(all_ret[k], 1) for k in all_ret}
    return all_ret


def render(H=None, W=None, focal=None, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., sp_input=None, tp_input=None, use_viewdirs=False, **kwargs):
    """Render rays
    """
    rays_o, rays_d = rays[:,0,...], rays[:,1,...]

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [rays_d.shape[0],-1,3]).float()

    sh = rays_d.shape # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [sh[0],-1,3]).float()
    rays_d = torch.reshape(rays_d, [sh[0],-1,3]).float()
    near = torch.reshape(near, [sh[0],-1,1]).float()
    far = torch.reshape(far, [sh[0],-1,1]).float()

    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, sp_input=sp_input, tp_input=tp_input, **kwargs)
    # for k in all_ret:
    for k in ['rgb_map', 'disp_map', 'acc_map', 'pts_mask', 'raw']:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[2:])
        # k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def test(chunk, render_kwargs, savedir=None, test_more=False):
    total_psnr = 0.0
    num = 0
    batch_size = global_args.batch_size
    poses_num = batch_size
    ratio = global_args.image_scaling
    
    # test_novel_pose = (not global_args.save_weights)# or test_more
    if test_more:
        poses_num = 20 # 8
        batch_size = 1
        # render_kwargs['network_fn'].module.correction_field = False

    if global_args.data_set_type in ["H36M_B", "H36M", "H36M_P", "H36M_B_All"]:
        data_interval = 20
        start_pose = 20
        test_set = H36MDatasetBatch(global_args.data_root, split=global_args.test_split, view_num=global_args.view_num,\
            # start=240, interval=20, poses_num=poses_num, border=global_args.border)
            start=start_pose, interval=data_interval, poses_num=poses_num, mean_shape=global_args.mean_shape, new_mask=global_args.new_mask)
        H,W = int(1000 * ratio), int(1000 * ratio)
        interval = 1
    
    if global_args.data_set_type in ["NeuBody_B"]:
        # for 313: start 0  interval 60
        # for 315: start 330 interval 60
        # for 377: start 180 interval 60
        # pose num 1 + 4 = 5
        # view interval 4 
        start_pose = 0 # 300
        # start_pose = 180 + 3*60 # 300
        data_interval = 60
        poses_num = 5
        # human_name = "human_377"
        human_name = os.path.basename(global_args.data_root)
        savedir = os.path.join(savedir, human_name)
        os.makedirs(savedir, exist_ok=True)
        # print(global_args.view_num)
        test_set = NeuBodyDatasetBatch(global_args.data_root, split=global_args.test_split, view_num=global_args.view_num,\
            start=start_pose, interval=data_interval, poses_num=poses_num, image_scaling=global_args.image_scaling) 
        H,W = int(1024 * ratio), int(1024 * ratio)
        interval = 4
    
    if global_args.data_set_type in ["THuman_B", "THuman", "THuman_P"]:
        data_interval = 1
        start_pose = 27
        test_set = THumanDatasetBatch(global_args.data_root, split=global_args.test_split, view_num=global_args.view_num,\
            start=start_pose, interval=data_interval, poses_num=poses_num, model=global_args.model, male=global_args.male, mean_shape=global_args.mean_shape) # hyd 13 lc 5
        H,W = int(512 * ratio), int(512 * ratio)
        interval = 12

    if global_args.data_set_type in ["THuman_B_R"]:
        test_set = THumanDatasetBatchRandom(global_args.data_root, split=global_args.test_split, view_num=global_args.view_num,\
            start=13, interval=1, poses_num=poses_num, male=global_args.male, mean_shape=global_args.mean_shape) # hyd 13 lc 5
        H,W = int(512 * ratio), int(512 * ratio)
        interval = 12
    
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)
    
    sp_input=None
    for i, data in enumerate(test_loader):
        data = to_cuda(device, data)
        if i==0:
            sp_input = data
        # sp_input = data
        tp_input = data

        if test_more and i==0 and global_args.data_set_type in ["THuman_B", "THuman_B_R", "THuman", "THuman_P"]: #, "NeuBody_B"]:
            interval = 1
        if test_more and i>0 and global_args.data_set_type in ["THuman_B", "THuman_B_R", "THuman", "THuman_P"]:
            interval = 12
        if test_more and i>0 and global_args.data_set_type in ["NeuBody_B"]:
            interval = 4
        # if test_more and i>0 and global_args.data_set_type in ["H36M_B", "H36M", "H36M_P", "H36M_B_All"]:
        #     interval = 10

        print("pose: ", i)
        for k in range(0, tp_input['rgb_all'].shape[1], interval):
            batch_rays = torch.stack([tp_input['ray_o_all'][:,k], tp_input['ray_d_all'][:,k]], 1) # (B,2,N_rand,3)
            near=tp_input['near_all'][:,k]
            far=tp_input['far_all'][:,k]
            target_s = tp_input['rgb_all'][:, k]
            msk = tp_input['msk_all'][:,k]
            mask_at_box = tp_input['mask_at_box_all'][:,k]

            if global_args.data_set_type in ["H36M_B", "H36M_P", "H36M_B_All"]:
                batch_rays = batch_rays.reshape(batch_size,2,H,W,3)[:, :, 150:662, 250:762, :].reshape(batch_size, 2, -1, 3)
                near = near.reshape(batch_size,H,W,1)[:, 150:662, 250:762, :].reshape(batch_size,-1,1)
                far = far.reshape(batch_size,H,W,1)[:, 150:662, 250:762, :].reshape(batch_size,-1,1)
                target_s = target_s.reshape(batch_size,H,W,3)[:, 150:662, 250:762, :].reshape(batch_size,512,512,3)
                msk = msk.reshape(batch_size,H,W,1)[:, 150:662, 250:762, :].reshape(batch_size,512,512)
                mask_at_box = mask_at_box.reshape(batch_size,H,W,1)[:, 150:662, 250:762, :].reshape(batch_size,512,512) 

            rgb, disp, acc, extras = render(chunk=chunk, rays=batch_rays, sp_input=sp_input, tp_input=tp_input,
                                            near=near, far=far, **render_kwargs)
            rgb = rgb.reshape(batch_size,512,512,3)
            target_s = target_s.reshape(batch_size,512,512,3)
            msk = msk.reshape(batch_size,512,512)
            mask_at_box = mask_at_box.reshape(batch_size,512,512)
            
            # if savedir is not None:
            for j in range(batch_size):
                img_pred = rgb[j]
                gt_img = target_s[j]
                
                # img_gt_all = tp_input['o_img_all'][:, k].cpu().numpy().transpose(0,2,3,1)
                # gt_img = img_gt_all[j]
                
                rgb8 = to8b(img_pred.cpu().numpy())
                gt_rgb8 = to8b(gt_img.cpu().numpy())
                rgb8 = np.concatenate([rgb8, gt_rgb8], axis=1)
                filename = os.path.join(savedir, '{:02d}_{:02d}_{:02d}.png'.format(int(sp_input['pose_index'][j]), int(tp_input['pose_index'][j]), k))
                imageio.imwrite(filename, rgb8)
            
                if k < 24:
                    # img_pred[msk[j]==100] = 0
                    # gt_img[msk[j]==100] = 0
                    # mask_at_box should be boolean type, not int (如果是1而不是False,会被当做坐标。此时用==1)
                    test_loss = img2mse(img_pred[mask_at_box[j]], gt_img[mask_at_box[j]])
                    psnr = round(mse2psnr(test_loss).item(), 3)
                    print("[Test] ", "Source:", int(sp_input['pose_index'][j]), " Target:", int(tp_input['pose_index'][j]), " View:", k, \
                        " Loss:", round(test_loss.item(), 5), \
                        " PSNR: ", {psnr})
                    total_psnr += mse2psnr(test_loss).item()
                    num += 1
                    if global_args.data_set_type in ["H36M_B", "H36M", "H36M_P", "H36M_B_All"]:
                        input_img = sp_input['img_all'][j].cpu().numpy()[..., 150:662, 250:762].transpose(2,0,3,1).reshape(512, -1, 3) * 255. #NCHW->HNWC
                    else:
                        input_img = sp_input['img_all'][j].cpu().numpy().transpose(2,0,3,1).reshape(512, -1, 3) * 255. #NCHW->HNWC
                    input_img = cv2.resize(input_img, (512*2, int(input_img.shape[0] * 512*2 / input_img.shape[1])))
                    # img = image_add_text(filename, 'PSNR: %f' % (psnr), 20, 20, text_color=(255, 255, 255), text_size=20)
                    img = rgb8
                    img = np.concatenate([input_img, img], axis=0)
                    imageio.imwrite(filename, to8b(img/255.))

                    gt_filename = os.path.join(savedir, 'frame{:04d}_view{:04d}_gt.png'.format(int(tp_input['pose_index'][j])*data_interval+start_pose, k))
                    pred_filename = os.path.join(savedir, 'frame{:04d}_view{:04d}.png'.format(int(tp_input['pose_index'][j])*data_interval+start_pose, k))
                    imageio.imwrite(gt_filename, to8b(gt_img.cpu().numpy()))
                    imageio.imwrite(pred_filename, to8b(img_pred.cpu().numpy()))
                else:
                    print("[Test] ", "Source:", int(sp_input['pose_index'][j]), " Target:", int(tp_input['pose_index'][j]), "View:", k)
    
    # render_kwargs['network_fn'].module.smooth_loss = pre
    avg_psnr = total_psnr / num
    np.save(savedir+'/psnr_{}.npy'.format(int(avg_psnr*100)), np.array(avg_psnr))

    if global_args.data_set_type in ["H36M_B", "H36M_P", "H36M_B_All"]:
        novel_pose = "_03.png"
    elif global_args.data_set_type in ["THuman_B"]:
        novel_pose = "_12.png"
    else:
        novel_pose = "_00.png"
    # novel_pose = "_01.png"
    images = [img for img in os.listdir(savedir) if img.endswith(novel_pose)]
    images.sort()
    images_to_video(savedir, video_name="novel_pose", images=images, fps=1)
    
    images = [img for img in os.listdir(savedir) if img.startswith("00_00_")]
    images.sort()
    images_to_video(savedir, video_name="novel_view", images=images, fps=3)

    return


def create_nerf(args, device=device):
    """Instantiate NeRF's MLP model.
    """
    Human_NeRF = return_model(global_args)
    model = Human_NeRF
    
    grad_vars = list(model.parameters())
    model_fine = None
    
    if args.N_importance > 0:
        model_fine = CorrectionByf3d()
        grad_vars += list(model_fine.parameters())
    
    network_query_fn = lambda inputs, viewdirs, network_fn, sp_input=None, tp_input=None : run_network(inputs, 
                                                            viewdirs, network_fn, sp_input=sp_input, tp_input=tp_input)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [os.path.join(basedir, expname, args.ft_path)]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if '.tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    if global_args.ddp:
        local_rank = global_args.local_rank
        model = model.to(local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True).to(device)
    else:
        model = nn.DataParallel(model).to(device)
    
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'N_importance' : args.N_importance,
        # 'white_bkgd' : args.white_bkgd,
        # 'network_fine' : model_fine,
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    """
    if not global_args.occupancy:
        raw2alpha = lambda raw, dists, act_fn=density_actfn: 1.-torch.exp(-act_fn(raw)*dists)
        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
        rgb = rgb_actfn(raw[...,:3])  # [N_rays, N_samples, 3]
        # rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        noise = 0.
        alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], alpha.shape[1], 1)), 1.-alpha + 1e-10], -1), -1)[:, :, :-1]
        # weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    else:
        rgb = rgb_actfn(raw[...,:3])  # [N_rays, N_samples, 3]
        alpha = rgb_actfn(raw[...,3])  # [N_rays, N_samples]
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], alpha.shape[1], 1)), 1.-alpha + 1e-10], -1), -1)[:, :, :-1]
    
    T_s = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], alpha.shape[1], 1)), 1.-alpha + 1e-10], -1), -1)[:, :, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        print("Using White Bkgd!!")
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map, T_s


def render_rays(ray_batch, network_fn, network_query_fn, N_samples,
                perturb=0., N_importance=0, network_fine=None, white_bkgd=False,
                sp_input=None, tp_input=None):

    # N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, :, 0:3], ray_batch[:, :, 3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [ray_batch.shape[0],-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    z_vals = near * (1.-t_vals) + far * (t_vals)
    # z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    raw, other_loss = network_query_fn(pts, viewdirs, network_fn, sp_input=sp_input, tp_input=tp_input)
    # smpl_query_pts, smpl_src_pts = raw[:, :, :, 8:11], raw[:, :, :, 11:14]
    # raw, pts_mask, correction = raw[:, :, :, 0:4], raw[:, :, :, 4:5], raw[:, :, :, 5:8]
    smpl_query_pts, smpl_src_pts = raw[:, :, :, 11:14], raw[:, :, :, 14:17]
    raw, pts_mask, correction, correction_ = raw[:, :, :, 0:4], raw[:, :, :, 4:5], raw[:, :, :, 5:8], raw[:, :, :, 8:11]
    rgb_map, disp_map, acc_map, weights, depth_map, T_s = raw2outputs(raw, z_vals, rays_d, white_bkgd)


    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    ret['smpl_query_pts'] = smpl_query_pts
    ret['smpl_src_pts'] = smpl_src_pts
    # ret['T_s'] = T_s
    ret['correction_'] = correction_
    ret['other_loss'] = other_loss
    ret['correction'] = correction
    # ret['correction_'] = correction
    ret['pts_mask'] = pts_mask
    ret['raw'] = raw

    return ret


def train(nprocs, global_args):
    args = global_args
    training_set = return_dataset(global_args)

    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(training_set)
        training_loader = DataLoader(training_set, batch_size=1, num_workers=0, sampler=train_sampler)
    else:
        training_loader = DataLoader(training_set, batch_size=global_args.batch_size, shuffle=True, num_workers=global_args.num_worker, pin_memory=False)
    
    # Multi-GPU
    args.n_gpus = torch.cuda.device_count()
    print("Using {} GPU(s).".format(args.n_gpus))

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start
    N_iters = global_args.N_iteration + 1

    # Summary writers
    # global writer
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    if start % 10 != 0:
        start = start + 1
    render_kwargs_train['network_fn'].train()
    scaler = GradScaler(enabled=False)
    skip_step = len(training_set.train_view)
    iter_per_epoch = skip_step * len(training_loader)
    
    running_loss = 0.0
    running_img_loss = 0.0
    running_acc_loss = 0.0
    running_cor_loss = 0.0
    running_density_loss = 0.0
    running_normal_smooth_loss = 0.0
    running_cor_smooth_loss = 0.0
    running_smpl_normal_loss = 0.0

    if global_args.save_weights == 0:
        print("Begin to test, save_weights == 0")
        testsavedir = os.path.join(basedir, expname, 'testset_{:06d}_more_real_ssim_psnr'.format(global_step))
        os.makedirs(testsavedir, exist_ok=True)
        render_kwargs_train['network_fn'].eval()
        with torch.no_grad():
            if global_args.data_set_type in ["H36M_B", "H36M", "H36M_P"]:
                # test(args.chunk, render_kwargs_train, savedir=testsavedir, test_more=True)
                test_H36M(args.chunk, render_kwargs_train, savedir=testsavedir, global_args=global_args, device=device, render=render, test_persons=global_args.test_persons)
            elif global_args.data_set_type in ["THuman_B"]:
                test_THuman_ssim(args.chunk, render_kwargs_train, savedir=testsavedir, global_args=global_args, device=device, render=render)
                # test_THuman(args.chunk, render_kwargs_train, savedir=testsavedir, global_args=global_args, device=device, render=render)
                # test(args.chunk, render_kwargs_train, savedir=testsavedir, test_more=True)
            else:
                test(args.chunk, render_kwargs_train, savedir=testsavedir, test_more=True)
                
        render_kwargs_train['network_fn'].train()
        print('Saved test set')
        global_args.save_weights = 1

        return
    
    # for global_step in range(start, N_iters, skip_step * len(training_loader)):
    while global_step < N_iters:

        epoch = global_step // iter_per_epoch
        if args.ddp:
            training_loader.sampler.set_epoch(epoch)

        for i, data in enumerate(training_loader):
            sp_input = data if not global_args.data_set_type in ["H36M_P", "THuman_P"] else data['sp_input']
            tp_input = data if not global_args.data_set_type in ["H36M_P", "THuman_P"] else data['tp_input']
            sp_input, tp_input = to_cuda(device, sp_input, tp_input)
            time0 = time.time()

            for k in range(0, tp_input['rgb_all'].shape[1]):
            # for k in range(0, len(tp_input['rgb_all'])):
                with autocast(enabled=False):
                    batch_rays = torch.stack([tp_input['ray_o_all'][:,k], tp_input['ray_d_all'][:,k]], 1) # (B,2,N_rand,3)
                    near=tp_input['near_all'][:,k]
                    far=tp_input['far_all'][:,k]
                    target_s = tp_input['rgb_all'][:, k]
                    bkgd_msk = tp_input['bkgd_msk_all'][:, k]
                    sp_input['global_step'] = torch.ones_like(sp_input['pose_index']) * (global_step + k)
                    sp_input['smooth_interval'] = torch.ones_like(sp_input['pose_index']) * global_args.smooth_interval
                    
                    ###  Core optimization loop  ###
                    rgb, _, acc, extras = render(chunk=args.chunk, rays=batch_rays, sp_input=sp_input, tp_input=tp_input,
                                                    near=near, far=far, **render_kwargs_train)
                    
                    ### calc loss ###
                    # pose_flag = (sp_input['pose_index']==tp_input['pose_index'])
                    img_loss = img2mse(rgb, target_s) if not global_args.data_set_type in ["H36M_P", "THuman_P"] else extended_img2mse(rgb, target_s, sp_input['pose_index'], tp_input['pose_index'])
                    acc_loss = img2mse(bkgd_msk.squeeze(2), acc) if (global_args.acc_loss and (not global_args.half_acc)) else torch.tensor(0.)
                    pts_mask = extras['pts_mask'].squeeze(-1)==1
                    correction_loss = (img2mse(extras['correction'][pts_mask], 0)+img2mse(extras['correction_'][pts_mask], 0)) if (global_args.correction_loss) else torch.tensor(0.)
                    consistency_loss = torch.tensor(0.) if (not global_args.consistency_loss) else (img2mse(extras['smpl_query_pts'][pts_mask], extras['smpl_src_pts'][pts_mask]))
                    density_loss = torch.tensor(0.) if not global_args.density_loss else 0.005 * F.l1_loss(torch.exp((-1.0)*density_actfn(extras['raw'][pts_mask][..., -1])), torch.ones_like(extras['raw'][pts_mask][..., -1])) 
                    loss = img_loss + correction_loss + acc_loss + consistency_loss + extras['other_loss'][0][0] + density_loss
                    
                    running_loss += loss.item() if img_loss!=0. else (loss.item() + img2mse(rgb, target_s).item())
                    running_img_loss += img_loss.item() if img_loss!=0. else img2mse(rgb, target_s).item()
                    running_acc_loss += acc_loss.item()
                    running_density_loss += density_loss.item()
                    running_cor_loss += correction_loss.item()
                    running_normal_smooth_loss += extras['other_loss'][0][1].item()
                    running_cor_smooth_loss += extras['other_loss'][0][2].item()
                    running_smpl_normal_loss += extras['other_loss'][0][3].item()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # loss.backward()
                # optimizer.step()
                optimizer.zero_grad()

            ###   update learning rate   ###
            decay_rate = 0.5
            decay_steps = global_args.decay_steps
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            ###     Logger    ###
            dt = (time.time()-time0) / skip_step
            global_step = skip_step + global_step
            if (global_step) % args.i_print == 0 and global_step > 1:
                psnr = round(mse2psnr(torch.tensor(running_img_loss / args.i_print)).item(), 3)
                writer.add_scalar("All training loss", running_loss / args.i_print, global_step)
                writer.add_scalar("image loss", running_img_loss / args.i_print, global_step)
                writer.add_scalar("acc weights loss", running_acc_loss / args.i_print, global_step)
                writer.add_scalar("density loss", running_density_loss / args.i_print, global_step)
                writer.add_scalar("correction loss", running_cor_loss / args.i_print, global_step)
                writer.add_scalar("correction smooth loss", running_cor_smooth_loss / args.i_print, global_step)
                writer.add_scalar("surface normal smooth loss", running_normal_smooth_loss / args.i_print * args.smooth_interval, global_step)
                writer.add_scalar("smpl normal loss", running_smpl_normal_loss / args.i_print * args.smooth_interval, global_step)
                writer.add_scalar("psnr", psnr, global_step)
                print("[TRAIN] Epoch:{}  Iter: {}  Loss: {} PSNR: {}  Time: {} s/iter".format(epoch, global_step, round(running_loss / args.i_print, 5), psnr, round(dt, 3)))
                running_loss = 0.0
                running_img_loss = 0.0
                running_acc_loss = 0.0
                running_cor_loss = 0.0
                running_normal_smooth_loss = 0.0
                running_density_loss = 0.0
                running_cor_smooth_loss = 0.0
                running_smpl_normal_loss = 0.0
            
            if (global_step)%args.i_weights == 0 and global_step > 1 and global_args.save_weights:
                path = os.path.join(basedir, expname, '{:06d}.tar'.format(global_step))
                
                if (args.ddp and dist.get_rank() == 0) or not args.ddp:
                    # torch.save(model.module.state_dict(), "%d.ckpt" % epoch)
                    torch.save({
                        'global_step': global_step,
                        'network_fn_state_dict': render_kwargs_train['network_fn'].module.state_dict(),
                        # 'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                        # 'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
                    print('Saved checkpoints at', path)
            
            # print(global_step, args.i_testset, args.i_weights)
            if (global_step%args.i_testset) == 0 and global_step > 1:
                testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(global_step))
                # os.makedirs(testsavedir, exist_ok=True)
                render_kwargs_train['network_fn'].eval()

                with torch.no_grad():
                    if  (global_step)%(args.i_weights)==0:

                        testsavedir = testsavedir + '_more'
                        os.makedirs(testsavedir, exist_ok=True)

                        if global_args.data_set_type in ["H36M_B", "H36M", "H36M_P"]:
                            test_H36M(args.chunk, render_kwargs_train, savedir=testsavedir, global_args=global_args, device=device, render=render, test_persons=global_args.test_persons)
                        
                        if global_args.data_set_type in ["THuman_B"]:
                            if global_step >= (args.N_iteration-1000):
                            # if global_step >= 119000:
                                # test_THuman(args.chunk, render_kwargs_train, savedir=testsavedir, global_args=global_args, device=device, render=render)
                                testsavedir = testsavedir + '_final_evaluation'
                                os.makedirs(testsavedir, exist_ok=True)
                                test_THuman_ssim(args.chunk, render_kwargs_train, savedir=testsavedir, global_args=global_args, device=device, render=render)
                            else:
                                test(args.chunk, render_kwargs_train, savedir=testsavedir, test_more=True)
                        
                        if global_args.data_set_type in ["NeuBody_B"]:
                            test(args.chunk, render_kwargs_train, savedir=testsavedir, test_more=True)
                    
                render_kwargs_train['network_fn'].train()
                print('Saved test set')
    # DDP:  
    # cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print_args(global_args)
    torch.multiprocessing.set_start_method('spawn', force=True)
    train(torch.cuda.device_count(), global_args)
