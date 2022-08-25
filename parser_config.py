import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')

    # training options
    parser.add_argument("--N_rand", type=int, default=1024*32, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--decay_steps", type=int, default=10000, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*64, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk_per_gpu", type=int, default=1024*64*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--with_viewdirs", type=int, default=1, 
                        help='use full 5D input instead of 3D')

    # dataset options
    parser.add_argument("--data_root", type=str, default='msra_h36m/S9/Posing', 
                        help='Dataset root dir')
    parser.add_argument("--data_set_type", type=str, default='multi_pair', 
                        help='Dataset root dir')
    parser.add_argument("--train_split", type=str, default="test", 
                        help='training dataloader type, choose whole image or random sample')
    parser.add_argument("--test_split", type=str, default="test", 
                        help='test dataloader type, choose whole image or random sample')
    parser.add_argument("--image_scaling", type=float, default="0.4", 
                        help='down sample factor')
    parser.add_argument("--model", type=str, default="correction_by_f3d", 
                        help='test dataloader type, choose whole image or random sample')
    parser.add_argument("--N_iteration",   type=int, default=48001, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    
    parser.add_argument("--use_os_env", type=int, default=0, help="--- ")
    parser.add_argument("--multi_person", type=int, default=1, help="--- ")
    
    parser.add_argument("--density_loss", type=int, default=0, help="--- ")
    parser.add_argument("--correction_loss", type=int, default=0, help="--- ")
    parser.add_argument("--acc_loss", type=int, default=1, help="--- ")
    parser.add_argument("--T_loss", type=int, default=1, help="--- ")
    parser.add_argument("--smooth_loss", type=int, default=1, help="--- ")
    parser.add_argument("--consistency_loss", type=int, default=0, help="--- ")
    
    parser.add_argument("--half_acc", type=int, default=0, help="--- ")
    parser.add_argument("--human_sample", type=int, default=0, help="--- ")
    parser.add_argument("--num_worker", type=int, default=8, help="--- ")
    parser.add_argument("--start", type=int, default=0, help="--- ")
    parser.add_argument("--interval", type=int, default=10, help="--- ")
    parser.add_argument("--poses_num", type=int, default=100, help="--- ")
    parser.add_argument("--num_instance", type=int, default=100, help="--- ")
    parser.add_argument("--test_num_instance", type=int, default=1, help="--- ")
    parser.add_argument("--random_pair", type=int, default=1, help="--- ")
    
    parser.add_argument("--use_f2d", type=int, default=0, help="--- ")
    parser.add_argument("--use_trans", type=int, default=0, help="--- ")
    parser.add_argument("--save_weights", type=int, default=1, help="--- ")
    parser.add_argument("--view_num",   type=int, default=3,  help='num of view, 3 for training, 4 for test')
    parser.add_argument("--border",   type=int, default=5, help='num of view, 3 for training, 4 for test')
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--ddp", type=int, default=0)
    parser.add_argument("--occupancy", type=int, default=0)
    parser.add_argument("--mean_shape", type=int, default=1)
    parser.add_argument("--correction_field", type=int, default=0)
    parser.add_argument("--skinning_field", type=int, default=0)
    parser.add_argument("--smooth_interval", type=int, default=4)
    parser.add_argument("--append_rgb", type=int, default=1)
    parser.add_argument("--male", type=int, default=0)
    parser.add_argument("--new_mask", type=int, default=0)
    parser.add_argument("--test_persons", type=int, default=2)
    parser.add_argument("--ani_nerf_ft", type=int, default=0)
    
    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=120, help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=12000, help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=3000, help='frequency of testset saving')

    parser.add_argument("--smpl_shape_loss", type=int, default=1)


    return parser


def print_args(args):
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')