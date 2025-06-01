import os
import sys
sys.path.append("./submodules")
sys.path.append("./submodules/GET3D")

import numpy as np
import random
import torch
import json
import argparse
from tracker_helper import get_object_model
from datetime import datetime

from src.tracking.data_formats import WaymoTracking, nuScenesTracking
from src.tracking.tracker import Tracker, DEVICE, ALL_CAMERAS_WAYMO, ALL_CAMERAS_NUSCENES
from devkit.convert_labels import WaymoDetectionLoader, nuScenesDetectionLoader

from src.tracking.write_waymo_out import write_scene_waymo
from src.tracking.write_nuscenes_out import write_scene_nuscenes

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(args, config):
    set_seed(2)
    data_kwargs = config["data_kwargs"]
    G_kwargs = config["G_kwargs"]
    G_kwargs['device'] = DEVICE
    optim_kwargs = config["optim_kwargs"]
    dataset = data_kwargs['dataset']
    split = data_kwargs['split']
    common_kwargs = config["common_kwargs"]
    
    print("Run tracker")
    path = args.tracking_pth
    detection_path = args.detections
    
    render_factor = optim_kwargs["resize_factor"]
    sample_factor = np.max([np.min([optim_kwargs["sample_factor"], 1.]), 0.05])
    
    weight_embedding    = args.weight_embedding
    weight_iou          = args.weight_iou
    weight_center       = args.weight_center
    affinity_threshold  = args.affinity_threshold
    maximum_distance    = args.maximum_distance
    max_lost            = args.max_lost
    
    # Set Storage
    dataset_base_pth = os.path.join(args.save_dir, dataset)
    if not os.path.exists(dataset_base_pth):
        os.makedirs(dataset_base_pth, exist_ok=True)

    # Create output Directories
    base_pth = os.path.join(dataset_base_pth, split +'_GET3D')
    if not os.path.exists(base_pth):
        os.makedirs(base_pth, exist_ok=True)

    no_viz = not args.viz

    # Dataset and Frame renderer
    if dataset == 'Waymo':
        cameras = data_kwargs['cameras'].upper()
        datasplit = WaymoTracking(path, split=split)
        if cameras == 'ALL':
            camera_names = ALL_CAMERAS_WAYMO
            # camera_names = CAMS_USED
        else:
            camera_names = [cameras]

        if split == 'testing':
            store_filename = 'qd3dt_waymo_detections_stored_'
            detection_loader = WaymoDetectionLoader
            
    elif dataset == 'nuScenes':
        cameras = data_kwargs['cameras'].upper()
        print("Running scenes {} to {}".format(config['start_idx'], config['stop_idx']))
        datasplit = nuScenesTracking(dataroot=path, 
                                    detection_root=detection_path, 
                                    split=split, 
                                    verbose=True, 
                                    start_idx=config['start_idx'], 
                                    stop_idx=config['stop_idx'],
                                    threshold=args.det_threshold)
        # first_scene = datasplit[0]
        # first_frame = first_scene[0]
        if cameras == 'ALL':
            camera_names = ALL_CAMERAS_NUSCENES
            # camera_names = CAMS_USED
        else:
            camera_names = [cameras]
            
        if split == 'testing':
            store_filename = 'centertrack_nuscenes_detections'
            detection_loader = nuScenesDetectionLoader
    else:
        datasplit = WaymoTracking(path)
        camera_names = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_RIGHT', 'SIDE_LEFT']

    if optim_kwargs["num_chunks"] > 1:
        chunk_id = optim_kwargs["chunk_id"]
        num_scenes = len(datasplit.scenes)
        chunk_size = int(np.ceil(num_scenes / optim_kwargs["num_chunks"]))
        datasplit.scenes = sorted(datasplit.scenes)[(chunk_id*chunk_size):((chunk_id+1)*chunk_size)]
        print("Chunking...\n")
        print("Running: \n")
        print(datasplit.scenes)

    # Object mdel and code_book if applicable    
    model, code_book = get_object_model(args.generator_checkpoint, G_kwargs, common_kwargs)

    scene_id = config["scene_id"]
    print(len(datasplit))
    if scene_id >= len(datasplit):
        print("Scene ID out of range")
        return
    for scene in datasplit:
        if scene_id > 0:
            scene = datasplit[scene_id] # this is the scene!!
        print('Tracking for {}'.format(scene.name))
        
        # include information on start and end scene id
        out_pth = os.path.join(base_pth, f"{scene.name}")
        # Create output Directory
        if not os.path.exists(out_pth):
            os.makedirs(out_pth, exist_ok=True)

        # create output File
        output_file_txt = os.path.join(out_pth, 'output.txt')
        print('Creating {} for outputs'.format(output_file_txt))
        with open(output_file_txt, 'w') as f:
            f.write('')

        if split == 'testing' and dataset == 'nuScenes':
            recording = None
            scene_detection_loader = None
        elif split == 'testing' and dataset == 'Waymo':
            recording = scene.name.split('-')[1].split('_with_camera_labels')[0]
            scene_detection_loader = detection_loader(detection_path, precompiled=True, split='testing', store_filename=store_filename, recording=recording)
        else:
            scene_detection_loader = None
            recording = None
            split = 'validation'

        frame_0 = scene[0]
        # num_cams x 9 informations
        intrinsics = frame_0['intrinsics']
        img_sz = (frame_0['height'], frame_0['width'])
        # 4x4
        veh_init_pose = frame_0['veh_pose']
        
        # get w_space statistics
        if args.matching_normalize:  
            num_samples = 1e5
            
            # draw num_samples latent codes of size G_kwargs["z_dim"] from a standard normal distribution
            z_tex_sample = torch.randn(int(num_samples), G_kwargs["z_dim"], device=G_kwargs['device'])
            z_geo_sample = torch.randn(int(num_samples), G_kwargs["z_dim"], device=G_kwargs['device'])
            
            # generate w_tex and w_geo
            ws_tex_sample = model.generate_w_tex(z_tex_sample)[:, 0, :]
            ws_geo_sample = model.generate_w_geo(z_geo_sample)[:, 0, :]
            
            # compute mean
            ws_tex_mean = torch.mean(ws_tex_sample, dim=0)
            ws_geo_mean = torch.mean(ws_geo_sample, dim=0)
            
            # compute std dev
            ws_tex_std = torch.std(ws_tex_sample, dim=0)
            ws_geo_std = torch.std(ws_geo_sample, dim=0)
            print("Normalizing w_space in matching")
        else:
            ws_tex_mean = torch.zeros(G_kwargs["z_dim"], device=G_kwargs['device'])
            ws_geo_mean = torch.zeros(G_kwargs["z_dim"], device=G_kwargs['device'])  
            ws_tex_std = torch.ones(G_kwargs["z_dim"], device=G_kwargs['device'])
            ws_geo_std = torch.ones(G_kwargs["z_dim"], device=G_kwargs['device'])
            print("Not normalizing w_space in matching")

        tracker = Tracker(model, code_book,
                        intrinsics, img_sz, optim_kwargs, render_factor,
                        weight_embedding, weight_iou, weight_center, affinity_threshold, maximum_distance, max_lost,
                        sample_factor, camera_names, veh_init_pose, no_viz, split, scene_detection_loader, dataset, 
                        ws_tex_mean=ws_tex_mean, ws_geo_mean=ws_geo_mean, ws_tex_std=ws_tex_std, ws_geo_std=ws_geo_std, 
                        ema_decay=args.ema_decay)
        print("Running Tracking on scene {}.".format(scene.name))
        tracklets = tracker.track(scene, scene_id=hash(scene.name) % 10000, s_pth=out_pth, recording=recording) 
        
        if dataset == 'Waymo':
            write_scene_waymo(tracklets, scene, out_pth)
        elif dataset == 'nuScenes':
            write_scene_nuscenes(tracklets, scene, out_pth)
        else:
            print("Writing dataset {} not supported".format(dataset))
            exit()    


if __name__ == "__main__":
    # Params
    parser = argparse.ArgumentParser()
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # General Settings
    parser.add_argument("-c", "--config", default='./configs/configs_track/nuscenes_viz.json')
    parser.add_argument("-sd", "--save_dir", default='./nuscenes_viz')
    parser.add_argument("-rn", "--run_name", default='run_{}'.format(date_str))
    # Generator Checkpoint
    parser.add_argument("--generator_checkpoint", type=str, default="./ckpt/GET3D/shapenet_car.pt")
    # Tracking Dataset + Detections
    parser.add_argument("--tracking_pth", type=str, default='./data/nuscenes')
    parser.add_argument("--detections", type=str, default=None)
    # Visualize results
    parser.add_argument("--viz", action='store_true', default=False)
    
    parser.add_argument("--weight_embedding", type=float, default=0.4)
    parser.add_argument("--weight_iou", type=float, default=1.4)
    parser.add_argument("--weight_center", type=float, default=1.0)
    parser.add_argument("--affinity_threshold", type=float, default=0.48)
    parser.add_argument("--maximum_distance", type=float, default=8.5)
    parser.add_argument("--max_lost", type=int, default=7)
    parser.add_argument("--det_threshold", type=float, default=0.4)
        
    # normalize codes for matching
    parser.add_argument("--matching_normalize", action='store_true', default=False)
    parser.add_argument("--ema_decay", type=float, default=0.8)

    args, others = parser.parse_known_args()
    
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    main(args, config)
