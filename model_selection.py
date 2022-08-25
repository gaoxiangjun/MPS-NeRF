from lib.skinnning_batch import CorrectionBatchBlend, DirectDeform, SKinningBatch, CorrectionBatch
from lib.correction import CorrectionByf3d, CorrectionByUvhAgg
from lib.run_nerf_helpers import *
from parser_config import *
from lib.h36m_dataset import H36MDataset, H36MDatasetBatch, H36MDatasetBatchAll, H36MDatasetPair
from lib.THuman_dataset import THumanDataset, THumanDatasetBatch, THumanDatasetPair, THumanDatasetBatchRandom

def return_model(global_args):
    if global_args.model == 'correction_by_f3d':
        Human_NeRF = CorrectionByf3d(human_sample=global_args.human_sample,
                        density_loss=global_args.density_loss, with_viewdirs=global_args.with_viewdirs,
                        use_f2d=global_args.use_f2d, smooth_loss=global_args.smooth_loss, 
                        use_trans=global_args.use_trans) # coarse deform + delta x (delta x decided by query [pts, f3d, diff])
    elif global_args.model == 'correction_by_f3d_fagg':
        Human_NeRF = CorrectionByf3d(
                        use_agg=True, human_sample=global_args.human_sample,
                        density_loss=global_args.density_loss, with_viewdirs=global_args.with_viewdirs,
                        use_f2d=global_args.use_f2d, smooth_loss=global_args.smooth_loss, 
                        use_trans=global_args.use_trans
                        )
    elif global_args.model == 'skinning_batch':
        Human_NeRF = SKinningBatch(
                        human_sample=global_args.human_sample,
                        density_loss=global_args.density_loss, 
                        with_viewdirs=global_args.with_viewdirs, 
                        use_f2d=global_args.use_f2d,
                        use_trans=global_args.use_trans,
                        smooth_loss=global_args.smooth_loss,
                        num_instances=global_args.num_instance,
                        mean_shape=global_args.mean_shape,
                        correction_field=global_args.correction_field, 
                        skinning_field=global_args.skinning_field,
                        data_set_type=global_args.data_set_type,
                        append_rgb=global_args.append_rgb
                        )
    elif global_args.model == 'direct_deform':
        Human_NeRF = DirectDeform(
                        human_sample=global_args.human_sample,
                        density_loss=global_args.density_loss, 
                        with_viewdirs=global_args.with_viewdirs, 
                        use_f2d=global_args.use_f2d,
                        use_trans=global_args.use_trans,
                        smooth_loss=global_args.smooth_loss,
                        num_instances=global_args.num_instance,
                        mean_shape=global_args.mean_shape,
                        correction_field=global_args.correction_field, 
                        skinning_field=global_args.skinning_field,
                        data_set_type=global_args.data_set_type,
                        append_rgb=global_args.append_rgb
                        )
    elif global_args.model == 'correction_batch':
        Human_NeRF = CorrectionBatch(
                        human_sample=global_args.human_sample,
                        density_loss=global_args.density_loss, 
                        with_viewdirs=global_args.with_viewdirs, 
                        use_f2d=global_args.use_f2d,
                        use_trans=global_args.use_trans,
                        smooth_loss=global_args.smooth_loss,
                        num_instances=global_args.num_instance,
                        mean_shape=global_args.mean_shape,
                        correction_field=global_args.correction_field, 
                        skinning_field=global_args.skinning_field,
                        data_set_type=global_args.data_set_type,
                        append_rgb=global_args.append_rgb
                        ) # only coarse deformation
    elif global_args.model == 'correction_batch_blend':
        Human_NeRF = CorrectionBatchBlend(
                        human_sample=global_args.human_sample,
                        density_loss=global_args.density_loss, 
                        with_viewdirs=global_args.with_viewdirs, 
                        use_f2d=global_args.use_f2d,
                        use_trans=global_args.use_trans,
                        smooth_loss=global_args.smooth_loss,
                        num_instances=global_args.num_instance,
                        mean_shape=global_args.mean_shape,
                        correction_field=global_args.correction_field, 
                        skinning_field=global_args.skinning_field,
                        data_set_type=global_args.data_set_type,
                        append_rgb=global_args.append_rgb
                        ) # only coarse deformation
    
    else:
        Human_NeRF = CorrectionByUvhAgg() # coarse deform + delta x (delta x decided by query [pts, f3d, diff])
    
    return Human_NeRF


def return_dataset(global_args, pairs=None):
    if global_args.data_set_type=="THuman":
        training_set = THumanDataset(
            global_args.data_root, 
            split=global_args.train_split, 
            view_num=global_args.view_num, 
            N_rand=global_args.N_rand,
            multi_person=global_args.multi_person,
            num_instance=global_args.num_instance,
            image_scaling=global_args.image_scaling,
            start=global_args.start, 
            interval=global_args.interval, 
            poses_num=global_args.poses_num
        )
    elif global_args.data_set_type=="THuman_B":
        training_set = THumanDatasetBatch(
            global_args.data_root, 
            split=global_args.train_split, 
            view_num=global_args.view_num, 
            N_rand=global_args.N_rand,
            multi_person=global_args.multi_person,
            num_instance=global_args.num_instance,
            image_scaling=global_args.image_scaling,
            start=global_args.start, 
            interval=global_args.interval, 
            poses_num=global_args.poses_num,
            male=global_args.male,
            mean_shape=global_args.mean_shape,
            model=global_args.model
        )
    elif global_args.data_set_type=="THuman_P":
        training_set = THumanDatasetPair(
            global_args.data_root,
            split=global_args.train_split, 
            view_num=global_args.view_num, 
            border=global_args.border, 
            N_rand=global_args.N_rand,
            multi_person=global_args.multi_person,
            num_instance=global_args.num_instance,
            image_scaling=global_args.image_scaling,
            start=global_args.start, 
            interval=global_args.interval, 
            poses_num=global_args.poses_num,
            random_pair=global_args.random_pair,
            male=global_args.male,
            mean_shape=global_args.mean_shape
        )
    elif global_args.data_set_type=="THuman_B_R":
        training_set = THumanDatasetBatchRandom(
            global_args.data_root, 
            split=global_args.train_split, 
            view_num=global_args.view_num, 
            N_rand=global_args.N_rand,
            multi_person=global_args.multi_person,
            num_instance=global_args.num_instance,
            image_scaling=global_args.image_scaling,
            start=global_args.start, 
            interval=global_args.interval, 
            poses_num=global_args.poses_num,
            male=global_args.male,
            mean_shape=global_args.mean_shape
        )
    elif global_args.data_set_type=="H36M":
        training_set = H36MDataset(
            global_args.data_root, 
            split=global_args.train_split, 
            view_num=global_args.view_num, 
            border=global_args.border, 
            N_rand=global_args.N_rand,
            multi_person=global_args.multi_person,
            num_instance=global_args.num_instance,
            image_scaling=global_args.image_scaling,
            start=global_args.start, 
            interval=global_args.interval, 
            poses_num=global_args.poses_num
        )
    elif global_args.data_set_type=="H36M_B":
        training_set = H36MDatasetBatch(
            global_args.data_root, 
            split=global_args.train_split, 
            view_num=global_args.view_num, 
            border=global_args.border, 
            N_rand=global_args.N_rand,
            multi_person=global_args.multi_person,
            num_instance=global_args.num_instance,
            image_scaling=global_args.image_scaling,
            start=global_args.start, 
            interval=global_args.interval, 
            poses_num=global_args.poses_num,
            mean_shape=global_args.mean_shape, 
            new_mask=global_args.new_mask
        )
    elif global_args.data_set_type=="H36M_P":
        training_set = H36MDatasetPair(
            global_args.data_root,
            split=global_args.train_split, 
            view_num=global_args.view_num, 
            border=global_args.border, 
            N_rand=global_args.N_rand,
            multi_person=global_args.multi_person,
            num_instance=global_args.num_instance,
            image_scaling=global_args.image_scaling,
            start=global_args.start, 
            interval=global_args.interval, 
            poses_num=global_args.poses_num,
            random_pair=global_args.random_pair,
            mean_shape=global_args.mean_shape, 
            new_mask=global_args.new_mask,
            test_persons=global_args.test_persons
        )
    elif global_args.data_set_type=="H36M_B_All":
        training_set = H36MDatasetBatchAll(
            global_args.data_root, 
            split=global_args.train_split, 
            view_num=global_args.view_num, 
            border=global_args.border, 
            N_rand=global_args.N_rand,
            multi_person=global_args.multi_person,
            num_instance=global_args.num_instance,
            image_scaling=global_args.image_scaling,
            start=global_args.start, 
            interval=global_args.interval, 
            poses_num=global_args.poses_num
        )
    elif global_args.data_set_type=="NeuBody_B":
        training_set = NeuBodyDatasetBatch(
            global_args.data_root, 
            split=global_args.train_split, 
            view_num=global_args.view_num, 
            border=global_args.border, 
            N_rand=global_args.N_rand,
            multi_person=global_args.multi_person,
            num_instance=global_args.num_instance,
            image_scaling=global_args.image_scaling,
            start=global_args.start, 
            interval=global_args.interval, 
            poses_num=global_args.poses_num
        )
    
    
    else:
        pass
    return training_set