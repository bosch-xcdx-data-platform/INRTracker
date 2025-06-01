import os
import operator
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from PIL import Image
import tqdm
import time
import plotly.express as px

from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles
from pytorch3d.transforms import so3_exp_map
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle
from pytorch3d.ops import box3d_overlap
from pytorch3d.loss.chamfer import chamfer_distance

from src.tracking.forward_renderer import render_texSDF_frame, get_waymo_base_cameras, _get_ray_sampler, render_get3D_frame
from src.tracking.optimizer import get_optim_state, compute_loss, get_params_optimizer, get_patch, apply_truncation_trick, move_tracklets_to_device
from src.tracking.utils import get_obj_in_cam,  get_tracklet_size, invert_pose, get_3d_box_corners, \
    get_2d_box_size_waymo, remove_duplicats_in_2D, get_cam_from_heading_wrt_cam, draw_3D_box, get_3D_box, kitti2vehicle

from src.tracking.kalman_filter import KF

# Check for rendering pipeline
# from src.texSDF.new_textured_SDF import new_texSDF


COLOR20_dict = {
    1 : (255, 0, 0),    # Red
    2 : (0, 128, 0),   # Green
    3 : (0, 0, 255),   # Blue
    4 : (255, 255, 0), # Yellow
    5 : (0, 255, 255), # Cyan
    6 : (255, 0, 255), # Magenta
    7 : (128, 0, 128), # Purple
    8 : (255, 165, 0), # Orange
    9 : (255, 192, 203), # Pink
    10 : (0, 128, 128), # Teal
    11 : (238, 130, 238), # Violet
    12 : (255, 191, 0), # Amber
    13 : (255, 0, 255), # Magenta
    0 : (191, 255, 0), # Lime
    15 : (0, 0, 128), # Navy
    16 : (128, 128, 0), # Olive
    17 : (128, 0, 0), # Maroon
    18 : (128, 128, 128), # Gray
    19 : (165, 42, 42), # Brown
    14 : (245, 245, 220), # Beige
}


ALL_CAMERAS_WAYMO = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_RIGHT', 'SIDE_LEFT']
ALL_CAMERAS_NUSCENES = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK_LEFT', 'BACK_RIGHT',  'BACK']
ALL_CAMERAS_KITTI = ['FRONT_LEFT']

# Waymo
# idx2cam = {0: 'FRONT', 1: 'FRONT_LEFT', 2: 'FRONT_RIGHT', 3: 'SIDE_LEFT', 4: 'SIDE_RIGHT'}
types_waymo = {0: 'UNKNOWN', 1: 'VEHICLE', 2: 'PEDESTRIAN', 3: 'SIGN', 4: 'CYCLIST'}

# Nuscenes
# idx2cam_nu = {0: 'FRONT', 1: 'FRONT_LEFT', 2: 'FRONT_RIGHT', 3: 'BACK_LEFT', 4: 'BACK_RIGHT', 5: 'BACK'}

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    print("USING GPU")
else:
    DEVICE = 'cpu'
    print("!!!! Running on CPU !!!!!")
    
class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.sum = 0
        self.n = 0
    
    def update(self, val):
        self.sum += val
        self.n += 1
    
    def get_avg(self):
        if self.n == 0:
            return 0  # To handle division by zero
        return self.sum / self.n
    
    def __str__(self):
        return f"AverageMeter: sum={self.sum}, n={self.n}, average={self.get_avg()}"

class Tracker(nn.Module):
    def __init__(self,
                 model,
                 code_book,
                 intrinsics,
                 img_sz, optim_kwargs,
                 render_factor=4.,
                 weight_embedding=0.6,
                 weight_iou=0.4,
                 weight_center=0.0,
                 affinity_threshold=0.3,
                 maximum_distance=12.,
                 max_lost=7,
                 sample_factor=1.,
                 camera_names=ALL_CAMERAS_WAYMO,
                 veh_init_pose=torch.eye(4),
                 no_viz=True,
                 split='validation',
                 scene_detection_loader=None,
                 data_set='nuScenes',
                 id_init=1,
                 log_dir='./tensorboard',
                 ws_tex_mean=None, ws_geo_mean=None, ws_tex_std=None, ws_geo_std=None, ema_decay=0.999):
        super(Tracker, self).__init__()
        # Add tensorboard writer
        self.log_dir = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        # self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Dataset
        self.split = split
        self.detections = scene_detection_loader
        self.data_set = data_set
        self.log_2D_bbox = True
        self.ema_decay = ema_decay
        print("ema_decay: ", self.ema_decay)
        
        self.idx2cam = {0: 'FRONT', 1: 'FRONT_LEFT', 2: 'FRONT_RIGHT', 3: 'SIDE_LEFT', 4: 'SIDE_RIGHT'} if data_set == "Waymo" \
        else {0: 'FRONT', 1: 'FRONT_LEFT', 2: 'FRONT_RIGHT', 3: 'BACK_LEFT', 4: 'BACK_RIGHT', 5: 'BACK'}

        # Dataset specific settings
        if data_set == 'Waymo':
            # Waymo
            self.cam2idx = {'FRONT': 0, 'FRONT_LEFT': 1, 'FRONT_RIGHT': 2, 'SIDE_LEFT': 3, 'SIDE_RIGHT': 4}
            if not all([c in self.cam2idx for c in camera_names]):
                camera_names = ALL_CAMERAS_WAYMO
            self.img_rate = 12 # Hz = 1/s 
                
        elif data_set == 'nuScenes':
            # TODO: Verify nuScenes naming
            self.cam2idx = {"FRONT": 0, "FRONT_LEFT": 1, "FRONT_RIGHT": 2, "BACK_LEFT": 3, "BACK_RIGHT": 4, "BACK": 5}
            if not all([c in self.cam2idx for c in camera_names]):
                camera_names = ALL_CAMERAS_NUSCENES
            self.img_rate = 12 # Hz = 1/s    
        else:
            # KITTI
            self.cam2idx = {'FRONT_LEFT': 0}
            camera_names = ALL_CAMERAS_KITTI

        # Initialize model
        self.model = model
        if code_book is not None:
            self.model_codes = code_book
            self.num_known_codes = len(code_book['sdf'])
        
        # Initialize motion model
        self.motionModel = KF

        # Render Settings
        self.sample_factor = sample_factor
        self.render_factor = optim_kwargs["resize_factor"]
        self.camera_names = camera_names
        self.camera_intrinsics = intrinsics
        focal = intrinsics[:, 0:2]
        principal = intrinsics[:, 2:4]
        img_h = img_sz[0]
        img_w = img_sz[1]
        self.cameras = get_waymo_base_cameras(focal, principal, img_w, img_h, factor=self.render_factor).to(DEVICE)
        self.halfFoV = torch.rad2deg(torch.atan2(self.cameras.image_size[:, 1] / 2, self.cameras.focal_length[:, 1])).detach().cpu()

        # counter
        self.trackers = []
        self.frame_count = 0
        self.id_count = id_init
        self.id_now_output = []
        self.active_tracklets = []
        self.colors = px.colors.qualitative.Light24

        self.detection_count = 0

        # self.ID_count[0] += 1
        # Vehicle model
        if data_set == 'Waymo':
            self.start_position = veh_init_pose
        else:
            self.start_position = np.eye(4)

        # Matching
        self.weight_embedding = weight_embedding
        self.weight_iou = weight_iou
        self.weight_center = weight_center
        self.affinity_threshold = affinity_threshold
        self.maximum_distance = maximum_distance

        self.scene_id = 0
        self.no_viz = no_viz
        self.viz_update = not no_viz
        self.max_lost = 7
        # Inverse Rendering Optimization
        self.optimize_w = optim_kwargs.get('optimize_w', False)
        self.lpips_kwargs = optim_kwargs.get('lpips_kwargs', {})
        self.tex_only_optim_adam = optim_kwargs.get('tex_only_optim_adam', True)
        self.optim_kwargs_tex_only = optim_kwargs.get('tex_only_optimizer_kwargs', {})
        if self.optimize_w:
            print("Optimizing objects in w-space")
        else:
            print("Optimizing objects in Z-space") 
        # TODO: Add to config file
        # Define optimization schedule for the matching step
        self.opti_config_detect = optim_kwargs['opti_config_detect']
        # self.lbfgs_k = optim_kwargs['lbfgs_k']
        # Define optimizer for the update step
        self.opti_config_update = optim_kwargs['opti_config_update']
        
        # matching normalize mean and std dev
        self.ws_tex_mean = ws_tex_mean
        self.ws_geo_mean = ws_geo_mean
        self.ws_tex_std = ws_tex_std
        self.ws_geo_std = ws_geo_std
        
        self.predict_avg_meter = AverageMeter()
        self.match_avg_meter = AverageMeter()
        self.match_non_rendering_avg_meter = AverageMeter()
        self.update_avg_meter = AverageMeter()
    
    def forward(self, scene, scene_id=0, s_pth=None, recording=None):
        self.scene_id = scene_id
        self.recording = recording

        # Init Tracklets in the first frame
        tracklets = {}
        DEBUG = False
        pred_ls = []

        first_frame = 0 # can make this arg, default 0 - change to arbitrary frame in the scene
        pbar = tqdm.tqdm(range(first_frame, len(scene)))
        for i in pbar:
            frame_time = time.time()
            # Get frame data
            frames = scene[i]
            veh_pose = frames['veh_pose']
            extrensics_i = frames['cam2veh']
            
            # Get detections
            detection_i = {i: {'lidar3D': frames['lidar_labels'], 'projected_lidar': frames['camera_labels']}}
            
            # Get images
            images_i = frames['imgs']
            
            # Transform to global coordinates to accomodate for different time stamps per camera
            if True and self.data_set == 'Waymo' or self.data_set == 'nuScenes':
                per_cam_veh_pose = frames['per_cam_veh_pose']
                local_veh_cam_pose = np.einsum('ij,cjk->cik', invert_pose(veh_pose), per_cam_veh_pose)
                extrensics_i = np.einsum('cij,cjk->cik', local_veh_cam_pose, extrensics_i)

            # Get camera pose from camera extrinsics (simply invert)
            pose_i = np.stack([invert_pose(ext) for ext in extrensics_i])

            if i > first_frame:
                # Predict location of all tracked objects in the current frame
                predict_time = time.time()
                self.predict(tracklets, veh_pose, prev_veh_pose, pose_i, frame_id=i)
                self.predict_avg_meter.update(time.time() - predict_time)
                self._set_update_description(pbar, frame_time, i, 'Predict')
            
            # Match detections to tracklets running inverse rendering on all detections and matching with reconstructed tracklets
            match_time = time.time()
            self.match(tracklets, detection_i, extrensics_i, images_i, frame_id=i, device=DEVICE, save_dir=s_pth)
            self.match_avg_meter.update(time.time() - match_time)
            self._set_update_description(pbar, match_time, i, 'Match')

            # if i > 0:
            update_time = time.time()
            self.update(tracklets, extrensics_i, veh_pose, images_i, frame_id=i, save_dir=s_pth)
            self.update_avg_meter.update(time.time() - update_time)
            self._set_update_description(pbar, update_time, i, 'Update')
            
            prev_veh_pose = frames['veh_pose']
            self.write_output(tracklets, frame_id=i, s_pth=s_pth)
 
        avg_meters = [self.predict_avg_meter, self.match_avg_meter, self.match_non_rendering_avg_meter, self.update_avg_meter]
        
        print("######### TIMING STATISTICS FOR ECCV REBUTTAL #########")
        for meter in avg_meters:
            print(meter)
        
        return tracklets
    
    def _set_update_description(self, pbar, start_time, frame_idx, step_name):
        description = f"frame: {int(frame_idx)} step: {step_name} - step forward: {time.time() - start_time:.4f} (s)"
        pbar.set_description(description)

    def track(self, scene, scene_id=0, s_pth=None, recording=None):
        return self.forward(scene, scene_id, s_pth, recording)

    ######################### PREDICTION STEP ########################################
    def predict(self, tracklets, veh_pose, prev_veh_pose, poses_fr, frame_id):
        removable_tracks = []
        
        # Use motion model to predict tracklet pose in next step
        for track_id in self.active_tracklets:
            track = tracklets[track_id]
            # if track[frame_id-1]['status'] == 'tracked':
            track['Filter'].kf.predict()
            # Transfer optimizable parameters from glob to the next step
            track[frame_id] = {k: v.clone()
                               for k, v in track['glob'].items()
                               if (isinstance(v, nn.Parameter) or isinstance(v, torch.Tensor))}
            # Freeze parameters in previous step
            track[frame_id-1].update({k: v.clone().detach().cpu()
                                      for k, v in track[frame_id-1].items()
                                      if (isinstance(v, nn.Parameter) or isinstance(v, torch.Tensor))})

            track[frame_id]['predict_state'] = {}
            x_global = track['Filter'].kf.x[:, 0]
            track[frame_id]['predict_state']['global_state'] = x_global

            x_veh = self.get_vehicle_state(x_global, veh_pose)
            track[frame_id]['predict_state']['state_veh'] = x_veh

            # Check Visibility in cameras and add predicted state in visible camera
            visible_in_cam_id = get_cam_from_heading_wrt_cam(x_veh, poses_fr, 2*self.halfFoV, tol=15)
            if visible_in_cam_id is None:
                # Out of frame predictions are removed from the set of active tracklets
                track[frame_id]['status'] = 'dead'
                removable_tracks.append(track['id'])
            else:
                # visible_in_cam_id = torch.argmax(viz_angle_in_cam)
                track[frame_id]['status'] = track[frame_id-1]['status']
                track[frame_id]['time_lost'] = track[frame_id - 1]['time_lost']

                shift_obj, rot_obj, scale_obj = \
                    get_obj_in_cam([{'c_x': x_veh[0],
                                         'c_y': x_veh[1],
                                         'c_z': x_veh[2],
                                         'heading': x_veh[3],
                                         'length': x_veh[4],
                                         'width': x_veh[5],
                                         'height': x_veh[6]}],
                                       poses_fr[visible_in_cam_id][None],
                                       self.camera_names[visible_in_cam_id])

                rot_cam = rot_obj.transpose(2, 1)
                # Convert rotation to logarithms
                rot_log = matrix_to_axis_angle(rot_cam)

                track[frame_id]['heading'].data = rot_log[:, 1].to(DEVICE)
                track[frame_id]['translation'].data = shift_obj[0].to(DEVICE)
                track[frame_id]['cam_id'] = int(visible_in_cam_id)

        for i in removable_tracks:
            self.active_tracklets.remove(i)
        return

    # for BEV visualization
    def get_bev_cam(self, extrensics_c):
        # extrensics_c is camera R and t in homogeneous coordinates with respect to the vehicle
        # extrensics_c_bev is extrensics_c rotated and translated and rotated to view the vehicle from a bird's eye view
        t = extrensics_c[:, :3, 3]
        R = extrensics_c[:, :3, :3]
        
        # translate to vehicle center
        t_bev = t.clone()
        t_bev[:, 0] = 0 # x
        t_bev[:, 1] = 0 # y
        t_bev[:, 2] = 0 # z
        
        # translate to 4 meters above the vehicle
        t_bev[:, 2] = t_bev[:, 2] + 4
        
        # rotate to point down towards the vehicle
        R_bev = R.clone()
        R_bev = torch.Tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).to(DEVICE) @ R_bev
        
        extrensics_c_bev = torch.eye(4).to(DEVICE)
        extrensics_c_bev[:, :3, 3] = t_bev
        extrensics_c_bev[:, :3, :3] = R_bev
        
        camera_pose_bev = self.get_veh2cam(extrensics_c_bev)
        
        return extrensics_c_bev, camera_pose_bev
        
    ######################### MATCHING STEP ########################################
    def match(self, tracklets, detections, extrensics, images, frame_id, device='cpu', save_dir=None):
        # Waymo specific
        obj_type = types_waymo
        cams = list(detections[frame_id]['projected_lidar'].keys())
        remove_duplicats_in_2D(detections, frame_id, cams, self.cameras, self.idx2cam, obj_type, self.render_factor) # Preprocess Waymo: No double 2D objects
        
        # Scaling and translation of get3D SDF models
        scale_correct_sdf = 1/0.45
        shift_correct = torch.tensor([0., 0., 0.])



        detections_3D_fitted = {}

        # In waymo/nuscenes iterate over all cameras
        
        # # for BEV visualization
        # self.camera_names = ["FRONT", "FRONT"]
        
        for cam_num, cam_name in enumerate(self.camera_names):
            ########### PRE-PROCESS DETECTIONS FOR RESPECTIVE CAMERA ############
            # TODO: Get detections separatly
            cam_id = self.cam2idx[cam_name]

            # Get detections before to decide on run using the test set
            if self.split == 'testing' and not self.recording is None:
                try:
                    detect_3d = self.detections.return_anno(recording=self.recording,
                                                            camera=cam_name.lower(),
                                                            image_frame=str(frame_id).zfill(7),
                                                            min_score=0.6,
                                                            max_depth=70)
                    if not len(detect_3d):
                        continue
                    
                    # Check for out of FOV objects in current camera
                    detect_3d = [det for det in detect_3d if
                                 not np.abs(np.rad2deg(np.arctan2(-det['c_x'], det['c_z']))) > self.halfFoV[cam_id] - .5]
                except:
                    print("Could not find annotations for {} Frame {} Camera {}".format(self.recording,str(frame_id).zfill(7),cam_name.lower()))
                    detect_3d = []
            else:
                if not cam_id in detections[frame_id]['projected_lidar']:
                    continue
                # Get 3D object detections for frame and camera
                object_types = ['VEHICLE']
                detect_keys = [k.split('_' + cam_name)[0] for k, v in detections[frame_id]['projected_lidar'][cam_id].items() if obj_type[v['type']] in object_types]
                if 'visible_cam' in list(detections[frame_id]['lidar3D'].values())[0]:
                    detect_3d = [detections[frame_id]['lidar3D'][k] for k in detect_keys if detections[frame_id]['lidar3D'][k]['visible_cam'] == cam_name]
                else:
                    detect_3d = [detections[frame_id]['lidar3D'][k] for k in detect_keys]
                
            if not len(detect_3d):
                continue

            # Sort detections by distance from camera
            d_sorted = np.argsort([np.linalg.norm(np.array([d['c_x'], d['c_y']])) for d in detect_3d])
            detect_3d = [detect_3d[dis_i] for dis_i in d_sorted]

            # Get camera poses and image size
            extrensics_c = extrensics[cam_id][None]
            camera_pose = self.get_veh2cam(extrensics_c)
            
            
            
            img_h = self.cameras[cam_id].image_size[:, 0].to(torch.int32)
            img_w = self.cameras[cam_id].image_size[:, 1].to(torch.int32)

            # Transform for detections from kitti fromat to vehicle centric format
            if self.data_set != 'nuScenes' and (self.split == 'testing' or self.data_set == 'KITTI'):
                detect_3d = kitti2vehicle(detect_3d, camera_pose)

            # Get observed image
            gt_img = images[cam_id]
            im = Image.fromarray(gt_img).resize((img_w, img_h))
            gt_img = torch.tensor(np.array(im, dtype=np.float32) / 255.)

            ########### OPTIMIZATION THROUGH INVERSE RENDERING OF ALL OBJECTS ############
            # Get camera wrt to each object
            obj_3d_detect_cam_i_dict = self.detection2tracklet(detect_3d,
                                                               camera_pose,
                                                               frame_id,
                                                               sdf_scale=scale_correct_sdf,
                                                               sdf_shift=shift_correct,
                                                               cam_id=cam_id,
                                                               cam_name=cam_name,
                                                               device=device)
            
            # If small objects result in 0 objects after detection2tracklet ignore camera image
            if not len(obj_3d_detect_cam_i_dict):
                continue            
            
            embedd_space ='w'if self.optimize_w else 'z'
            # Move tracklets to device
            obj_3d_detect_cam_i_dict = move_tracklets_to_device(obj_3d_detect_cam_i_dict, current_fr_id=frame_id, embedd_space=embedd_space, device=device)
            # Run optimization for detected objects
            obj_3d_detect_cam_i_dict, detect_optimizer = self.optimize_tracklet_on_image(obj_3d_detect_cam_i_dict,
                                                                                         gt_img, frame_id=frame_id,
                                                                                         cam_id=cam_id,
                                                                                         opti_config=self.opti_config_detect,
                                                                                         module_name="detect",
                                                                                         save_dir=save_dir)

            # Get state of object in vehicle coordinates given object in camera coordinates
            for d, (detect_key, detects) in enumerate(obj_3d_detect_cam_i_dict.items()):
                detects[frame_id]['state_veh'] = self.tracklet2detection_waymo(detects, camera_pose,
                                                                                frame_id, cam_id=cam_id)
                # detects[frame_id]['optimizer'] = detect_optimizer[d]

            detections_3D_fitted.update(obj_3d_detect_cam_i_dict)

        ########### MATCH DETECTIONS TO TRACKLETS ############
        match_non_rendering_time = time.time()
        avilable_tracklets = [(track_id, tracklets[track_id][frame_id]) for track_id in self.active_tracklets]
        fitted_detects = [fitted_detect[frame_id] for fitted_detect in detections_3D_fitted.values()]
        fitted_detects_keys = [det_key for det_key in detections_3D_fitted.keys()]
        
        # Match with and tracked tracklets
        if len(avilable_tracklets) > 0 and len(fitted_detects) > 0:
            # Compute Affinity for all detections and tracked tracklets
            score, box_iou, latent_score, center_distance, distance_score = self.compute_affinity(avilable_tracklets, fitted_detects)
            # Associate detections to tracklets given their affinity score
            matches, unmatched_dets, unmatched_tracklets, cost, aff_score = self.associate(score, center_distance, fitted_detects, avilable_tracklets,)
            tracklets_keep = [avilable_tracklets[track_i][0] for track_i in matches[:, 1]]
            # Preapre unmatched tracklets
            tracklets_unmatched = [avilable_tracklets[track_i][0] for track_i in unmatched_tracklets]
            # Prepare new detections to be added to the set of tracklets
            # tracklets_new_from_detect = [detect_i for detect_i in range(len(fitted_detects))
            #                              if not detect_i in list(matches[:, 0])]
            new_detect_keys = [fitted_detects_keys[detect_i] for detect_i in unmatched_dets]
        else:
            score = None
            # No tracklets available if restart from new detections
            # tracklets_new_from_detect = [int(key) for key in detections_3D_fitted.keys()]
            tracklets_keep = []
            # Mark al tracklets as unmatched if no detections are available
            tracklets_unmatched = [track[0] for track in avilable_tracklets]
            new_detect_keys = fitted_detects_keys

        # Set new tracklets
        # for detect_idx, (detect_key, new_detects) in enumerate(detections_3D_fitted.items()):
        for detect_key in new_detect_keys:
            new_detects = detections_3D_fitted[detect_key]
            # if int(detect_key) in tracklets_new_from_detect:
            # Give unique ID per scene
            new_id = int(self.id_count)
            self.id_count = 1 + self.id_count
            new_detects['id'] = new_id
            # Set track status, score and velocity
            new_detects[frame_id]['status'] = 'tracked'
            try:
                if not 'score' in detections[frame_id]['lidar3D'][detect_key]:
                    new_detects[frame_id]['score'] = 0.99999999
                else:
                    new_detects[frame_id]['score'] = detections[frame_id]['lidar3D'][detect_key]['score']
            except:
                new_detects[frame_id]['score'] = 0.99999999
            new_detects[frame_id]['new'] = True
            new_detects[frame_id]['velocity'] = torch.tensor([0., 0., 0.])
            # Define Box color for visualization
            new_detects['box_color'] = self.colors[np.mod(new_id, len(self.colors))]
            # Add to the set of all tracklets with unique ID
            tracklets.update({new_id: new_detects})
            # Add ID to active tracklets
            self.active_tracklets.append(new_id)

        # Keep old tracklets or set matched lost tracklets to tracked
        if len(tracklets_keep):
            for detect_idx, track_match_idx in matches:
                tracklet_idx = avilable_tracklets[track_match_idx][0]
                track_i = tracklets[tracklet_idx]
                # Copy data from detection to tracklet at current frame
                track_i[frame_id]['translation'].data = fitted_detects[detect_idx]['translation'].data
                track_i[frame_id]['heading'].data = fitted_detects[detect_idx]['heading'].data
                track_i[frame_id]['cam_id'] = int(fitted_detects[detect_idx]['cam_id'])
                track_i[frame_id]['state_veh'] = fitted_detects[detect_idx]['state_veh']
                track_i[frame_id]['{}_shape'.format(embedd_space)].data = fitted_detects[detect_idx]['{}_shape'.format(embedd_space)].data
                track_i[frame_id]['{}_tex'.format(embedd_space)].data = fitted_detects[detect_idx]['{}_tex'.format(embedd_space)].data
                if 'score' in fitted_detects[detect_idx]:
                    track_i[frame_id]['score'] = fitted_detects[detect_idx]['score']
                else:
                    if score is not None:
                        track_i[frame_id]['score'] = score[track_match_idx, detect_idx].cpu().detach().numpy()
                    else:
                        # Assign perfect score if runnind on annotations
                        track_i[frame_id]['score'] = 0.99999999
                # Set matched lost tracklets to tracked
                if track_i[frame_id-1]['status'] == 'lost':
                    track_i[frame_id]['status'] = 'tracked'
                    track_i[frame_id]['time_lost'] = 0

        # Add time step to lost tracklets and remove objects
        for tracklet_idx in tracklets_unmatched:
            track_i = tracklets[tracklet_idx]
            # Set lost tracklets to lost
            track_i[frame_id]['status'] = 'lost'
            track_i[frame_id]['time_lost'] += 1
            track_i[frame_id]['score'] = 0.1
            # Remove tracklets after k_steps or out of image
            if track_i[frame_id]['time_lost'] >= self.max_lost:
                self.active_tracklets.remove(tracklet_idx)
                track_i[frame_id]['status'] = 'dead'
                track_i[frame_id]['score'] = 0.
        self.match_non_rendering_avg_meter.update(time.time() - match_non_rendering_time)
        return tracklets

    ######################### UPDATE STEP ########################################
    def update(self, tracklets,  extrensics, veh_pose, images, frame_id, save_dir=None):
        embedd_space = 'w' if self.optimize_w else 'z'
        # Update global state of all tracklets
        update_tracklets = {k: val for k, val in tracklets.items()
                                 if (k in self.active_tracklets
                                     and not 'new' in val[frame_id]
                                     and val[frame_id]['status'] == 'tracked')}
        # Update Global state of existing and tracked objects
        for track_key, tracked_obj in update_tracklets.items():
            local_track = tracked_obj[frame_id]['state_veh']

            local_state = torch.cat([local_track['translation'],
                                     local_track['heading'][None],
                                     local_track['length'][None],
                                     local_track['width'][None],
                                     local_track['height'][None],
                                     torch.tensor([0., 0., 0.])])            
            # Get new state in global pose
            global_state = self.get_global_state(local_state, veh_pose)
            tracked_obj[frame_id]['global_state'] = global_state
            global_velocity = (tracked_obj[frame_id]['global_state'][:3] - tracked_obj[frame_id-1]['global_state'][:3])
            
            global_state[-3:] = global_velocity

            # Update motion model with new pose
            tracked_obj['Filter'].kf.update(global_state[:7])
            
            # Update w or z embedding codes
            local_embedd_shape =  tracked_obj[frame_id]['{}_shape'.format(embedd_space)].clone().data
            avg_embedd_shape = tracked_obj['glob']['{}_shape'.format(embedd_space)].clone().data
            local_embedd_tex =  tracked_obj[frame_id]['{}_tex'.format(embedd_space)].clone().data
            avg_embedd_tex = tracked_obj['glob']['{}_tex'.format(embedd_space)].clone().data
            new_avg_shape = avg_embedd_shape * self.ema_decay + (1 - self.ema_decay) * local_embedd_shape
            new_avg_tex = avg_embedd_tex * self.ema_decay + (1 - self.ema_decay) * local_embedd_tex
            
           
            
            tracked_obj['glob']['{}_shape'.format(embedd_space)].data = new_avg_shape
            tracked_obj['glob']['{}_tex'.format(embedd_space)].data = new_avg_tex
            
            
        #  Create global state for new objects
        new_tracklets = {k: val for k, val in tracklets.items()
                    if (k in self.active_tracklets
                        and 'new' in val[frame_id]
                        and val[frame_id]['status'] == 'tracked')}
        for track_key, tracked_obj in new_tracklets.items():
            track_id = tracked_obj["id"]
            # Get state for Kalman Filter: x, y, z, theta, l, w, h, dx, dy, dz
            local_track = tracked_obj[frame_id]['state_veh']
            local_state = torch.cat([local_track['translation'],
                                        local_track['heading'][None],
                                        local_track['length'][None],
                                        local_track['width'][None],
                                        local_track['height'][None],
                                        torch.tensor([0., 0., 0.])])

            global_state = self.get_global_state(local_state, veh_pose)
            tracked_obj[frame_id]['global_state'] = global_state
            tracked_obj['Filter'] = KF(global_state[:7], info=None, ID=track_id)
            tracked_obj['glob'] = {k: v.clone().detach()
                                for k, v in tracked_obj[frame_id].items()
                                if (isinstance(v, nn.Parameter) or isinstance(v, torch.Tensor))}
            
        # Set Global state for lost objects
        lost_tracklets = {k: val for k, val in tracklets.items()
                            if (k in self.active_tracklets
                                and not val[frame_id]['status'] == 'tracked'
                                and not val[frame_id]['status'] == 'dead')}
        
        for track_key, tracked_obj in lost_tracklets.items():
            tracked_obj[frame_id]['global_state'] = np.array(tracked_obj['Filter'].kf.x[:, 0], dtype=np.float32)
            
        with torch.no_grad():
            if self.viz_update:
                for cam_name in self.camera_names:
                    cam_id = self.cam2idx[cam_name]
                    camera_pose = self.get_veh2cam(extrensics[cam_id][None])

                    im = images[cam_id]
                    

                    update_tracklet = {k: val for k, val in tracklets.items()
                                        if k in self.active_tracklets
                                        and val[frame_id]['cam_id'] == cam_id}
                    # TODO: Get tracklets in the local vehicle frame from "tracked_obj[frame_id]['state_veh']" and convert with respect to the 
                    # respective camera and store as [tracklet_id][frame_id][key] with key in ['translation', 'heading', 'w_shape', 'w_tex', 'scale']
                    if len(update_tracklet) == 0:
                        gt_img = torch.tensor(gt_img, dtype=torch.float32, device=DEVICE) / 255.
                        vis_out = {"rgb_out": torch.zeros_like(gt_img), # pred rendering
                                    "gt_img": gt_img, # gt
                                    "instance_mask": torch.zeros_like(gt_img), # instance mask
                                    "im_w_bbox": gt_img, # 3D bbox
                                    "im_w_bbox_rgb_out": torch.zeros_like(gt_img), # 3D bbox,
                                    "im_overlay_bbox": gt_img, # 3D bbox + pred rendering overlay
                        }  
                        
                        vis_dir = os.path.join(save_dir, "vis", f"cam_{str(cam_id)}", f"frame_{str(frame_id)}")
                        os.makedirs(vis_dir, exist_ok=True)
                        for key, val in vis_out.items():
                            val_copy = val.clone().detach().cpu()
                            val_copy = np.array(val_copy.clamp(0,1) * 255., dtype=np.uint8)
                            val_copy = val_copy[0] if key == "depth" else val_copy                            
                            mode = "RGB" if key != "depth" else "L"
                            val_viz = val_copy
                            im = Image.fromarray(val_viz, mode=mode) 
                            os.makedirs(vis_dir, exist_ok=True)
                            vis_path = os.path.join(vis_dir, f"{key}.png")
                            im.save(vis_path)
                        continue

                    update_tracklet_cam_id = {}
                    
                    
                    for track_id, track_val in update_tracklet.items():
                        if "state_veh" not in track_val[frame_id]:
                            continue
                        
                        x_global = track_val["Filter"].kf.x[:, 0]
                        x_veh = self.get_vehicle_state(x_global, veh_pose)
                        
                        
                        translation_i, rot_obj, scale_obj = \
                            get_obj_in_cam([{'c_x': x_veh[0],
                                         'c_y': x_veh[1],
                                         'c_z': x_veh[2],
                                         'heading': x_veh[3],
                                         'length': x_veh[4],
                                         'width': x_veh[5],
                                         'height': x_veh[6]}],
                                       camera_pose,
                                       self.camera_names[cam_id])
                        
                        heading_i = track_val[frame_id]['heading'] # non smoother heading
                        #matrix_to_axis_angle(rot_obj.transpose(2, 1))[..., 1]
                        translation_i = translation_i.squeeze(0) # smoother translation
                        #track_val[frame_id]['translation'] 
                        
                        length_i = track_val['glob']['length'].clone()
                        width_i = track_val['glob']['width'].clone()
                        height_i = track_val['glob']['height'].clone()
                        scale_i = track_val['glob']['scale'].clone()
                        
                        update_tracklet_cam_id[track_id] = {frame_id: {}}
                        update_tracklet_cam_id[track_id][frame_id]['translation'] = translation_i.to(DEVICE) #track_val[frame_id]['translation']
                        update_tracklet_cam_id[track_id][frame_id]['heading'] = heading_i.to(DEVICE) # track_val[frame_id]['heading']
                        update_tracklet_cam_id[track_id][frame_id]['height'] = height_i.to(DEVICE)
                        update_tracklet_cam_id[track_id][frame_id]['length'] = length_i.to(DEVICE)
                        update_tracklet_cam_id[track_id][frame_id]['width'] = width_i.to(DEVICE)
                        update_tracklet_cam_id[track_id][frame_id]['scale'] = scale_i.to(DEVICE)
                        update_tracklet_cam_id[track_id][frame_id]['w_shape'] = track_val['glob']['w_shape'].clone()
                        update_tracklet_cam_id[track_id][frame_id]['w_tex'] = track_val['glob']['w_tex'].clone()
                    
                    if len(update_tracklet_cam_id) == 0:
                        continue
                    
                    rgb_out, mask_out, instance_mask, shape_code_i, tex_code_i, depth, lhw, mesh = render_get3D_frame(
                        update_tracklet_cam_id,
                        self.cameras[cam_id],
                        self.model,
                        current_fr_id=frame_id,
                        optimize_w=True,
                        n_per_step=6,
                        subsample=self.sample_factor,
                        device=DEVICE)

                    e_render = time.time()
                    
                    ###################### VISUALIZATION #################################
                    # Visualization Draw Bounding Boxes on Image
                    
                    if self.viz_update:    
                        gt_img = images[cam_id]
                    
                        # argument 'input' (position 1) must be Tensor, not numpy.ndarray
                        gt_img = torch.tensor(gt_img, dtype=torch.float32, device=DEVICE) / 255.
                        instances_colored = torch.zeros_like(gt_img)
                        grounded_instance_mask = instance_mask.clone() 
                        grounded_instance_mask[instance_mask >= len(COLOR20_dict)] = instance_mask[instance_mask >= len(COLOR20_dict)] % len(COLOR20_dict)
                        for color_idx, color_RGB in COLOR20_dict.items():
                            instances_colored[grounded_instance_mask == color_idx] = torch.tensor(color_RGB, dtype=torch.float32, device=DEVICE) / 255.
                        
                        # visualize_depth = depth - depth.min() / -depth.min() 
                        full_mask = mask_out[None, None].repeat(1,3,1,1).bool()
                        rgb_gt = gt_img.permute(2, 0,1)[None] # imp
                        rgb_overlay = rgb_gt.clone() # imp
                        
                        # rgb_overlay = (rgb_overlay * 0.6) + 1.0 * (1.0 - 0.6)    # fade background
                        fade_factor = 0.2 
                        rgb_overlay[full_mask] = (rgb_overlay[full_mask] * fade_factor) + (rgb_out.permute(2, 0,1)[None][full_mask].clamp(0.,1.) * (1.0 - fade_factor)) # adding rgb out overlay to gt
                        # rgb_overlay[full_mask] = rgb_out.permute(2, 0,1)[None][full_mask].clamp(0.,1.) # adding rgb out overlay to gt

                        result_cameras = self.cameras[cam_id].clone()
                        rgb_np = np.array(rgb_gt.permute(0,2,3,1)[0].clamp(0,1).cpu().clone().detach() * 255., dtype=np.uint8) # rgb_overlay
                        im = Image.fromarray(rgb_np)
                        # tracklet keys = obj_key, tracklet values = obj_i
                        # obj_i keys = frame_id, obj_i values = obj_i_frame, with heading, rotation, etc as keys
                        tracklet = {k: val for k, val in update_tracklet.items()
                                            if k in self.active_tracklets
                                            and val[frame_id]['cam_id'] == cam_id}
                        im_w_bbox = draw_3D_box(im, tracklet, frame_id, result_cameras)
                        
                        rgb_overlay_np = np.array(rgb_overlay.permute(0,2,3,1)[0].clamp(0,1).cpu().clone().detach() * 255., dtype=np.uint8)
                        im_overlay_bbox = Image.fromarray(rgb_overlay_np)
                        im_overlay_bbox = draw_3D_box(im_overlay_bbox, tracklet, frame_id, result_cameras)
                        
                        # rgb_out bbox
                        rgb_out_np = np.array(rgb_out.permute(2,0,1).clamp(0,1).cpu().clone().detach() * 255., dtype=np.uint8)
                        rgb_out_np = rgb_out_np.transpose(1,2,0)
                        im_rgb_out = Image.fromarray(rgb_out_np)
                        im_w_bbox_rgb_out = draw_3D_box(im_rgb_out, tracklet, frame_id, result_cameras)
                        # rgb_out bbox
                        
                        
                        # if k == 0:
                        #     im_first_setp = im_w_bbox.copy()
                            
                        vis_out = {"rgb_out": rgb_out, # pred rendering
                                    "gt_img": gt_img, # gt
                                    "instance_mask": instances_colored, # instance mask
                                    "im_w_bbox": im_w_bbox, # 3D bbox
                                    "im_w_bbox_rgb_out": im_w_bbox_rgb_out, # 3D bbox,
                                    "im_overlay_bbox": im_overlay_bbox, # 3D bbox + pred rendering overlay
                        }  # "depth": visualize_depth, # depth,
                        
                        
                        # only save on the last iteration of the optimization
                        if True: #k == n_opti_steps - 1:
                            vis_dir = os.path.join(save_dir, "vis", f"cam_{str(cam_id)}", f"frame_{str(frame_id)}")
                            os.makedirs(vis_dir, exist_ok=True)
                            for key, val in vis_out.items():
                                if key !=  "im_w_bbox" and key != "im_w_bbox_rgb_out" and key != "im_overlay_bbox":
                                    val_copy = val.clone().detach().cpu()
                                    val_copy = np.array(val_copy.clamp(0,1) * 255., dtype=np.uint8)
                                    val_copy = val_copy[0] if key == "depth" else val_copy                            
                                    mode = "RGB" if key != "depth" else "L"
                                    val_viz = val_copy
                                    im = Image.fromarray(val_viz, mode=mode) 
                                else:
                                    im = val
                                os.makedirs(vis_dir, exist_ok=True)
                                vis_path = os.path.join(vis_dir, f"{key}.png")
                                im.save(vis_path)
                        
                    ###################### VISUALIZATION #################################

           
        # Iterate through cameras
        if self.log_2D_bbox:
            for cam_name in self.camera_names:
                cam_id = self.cam2idx[cam_name]
            
                log_tracklets = {k: val for k, val in tracklets.items()
                                if (k in self.active_tracklets
                                    and val[frame_id]['cam_id'] == cam_id)}
               

                if len(log_tracklets):
                    extrensics_c = extrensics[cam_id][None]
                    camera_pose = self.get_veh2cam(extrensics_c)
                    rot_y2alpha = np.arctan2(camera_pose.T[0, 1, 0], camera_pose.T[0, 0, 0])

                    # Get alpha and 2D Bounding Box for KITTI eval before logging
                    result_cameras = self.cameras[cam_id].clone()

                    result_cameras.focal_length = result_cameras.focal_length * self.render_factor
                    result_cameras.principal_point = result_cameras.principal_point * self.render_factor
                    result_cameras.image_size = (result_cameras.image_size * self.render_factor).to(dtype=torch.int32)
                    im_h, im_w = result_cameras.image_size[0]

                    # box_center = get_3D_box(log_tracklets, frame_id, result_cameras, center=True)
                    box_corners = get_3D_box(log_tracklets, frame_id, result_cameras, center=False)

                    for k, (track_key, tracked_obj) in enumerate(log_tracklets.items()):
                        proj_3D_corners_k = box_corners[k]
                        # Left, Top, Right, Bottom
                        tracked_obj[frame_id]['bbox_2D'] = np.array([np.max([np.min(proj_3D_corners_k[..., 0], axis=0), 0]),
                                                                    np.max([np.min(proj_3D_corners_k[..., 1], axis=0), 0]),
                                                                    np.min([np.max(proj_3D_corners_k[..., 0], axis=0), int(im_w)]),
                                                                    np.min([np.max(proj_3D_corners_k[..., 1], axis=0), int(im_h)]),
                                                                    ], dtype=np.float32)
                        # TODO: Change alpha and rot_y from heading and check if heading and rot_y are betwee -pi and pi
                        tracked_obj[frame_id]['alpha'] = rot_y2alpha + tracked_obj[frame_id]['heading'][0].detach().cpu().numpy()
        return

    ######################### HELPER FUNCTIONS ########################################
    def optimize_tracklet_on_image(self, tracklet, gt_img, frame_id=0, cam_id=0, opti_config={}, module_name="update", save_dir=None):
        def _get_latent_sampled(num_code_samples):
            sampled_code_idx = np.random.permutation(np.arange(self.num_known_codes))[:num_code_samples]
            sampled_codes = {'z_' + key: list(operator.itemgetter(*sampled_code_idx)(list(code_type.values())))
                             for key, code_type in self.model_codes.items()}
            return sampled_codes

        # Get optimizer from visible objects
        optimizer_all, optimizer_all_tex, all_params_name = get_params_optimizer(tracklets=tracklet, 
                                                                                 frame=frame_id, 
                                                                                 optimize_w=self.optimize_w,
                                                                                 tex_only_optim_adam=self.tex_only_optim_adam,
                                                                                 optim_kwargs_tex_only=self.optim_kwargs_tex_only
                                                                                 )
        optimizer_all_tex = None

        # Prepare for loss functions
        # Random sample codes
        num_obj = len(tracklet)
        num_code_samples = np.max([num_obj * 8, 32])

        # # Get scale pf all objects
        detect_scales = [obj[frame_id]['scale'] for obj in tracklet.values()]

        # Volumetric loss
        num_sdf_samples = 800

        # Get sensor parameters
        img_h = self.cameras[cam_id].image_size[:, 0].to(torch.int32)
        img_w = self.cameras[cam_id].image_size[:, 1].to(torch.int32)
        aspect_ratio = img_h / img_w

        gt_img = gt_img.to(DEVICE)

        # Get raysampler
        raysampler = _get_ray_sampler(img_w, img_h, float(aspect_ratio))

        # Get codes to compare with
        try:
            sampled_codes = _get_latent_sampled(num_code_samples)
        except:
            sampled_codes = None

        if opti_config['early_stop'] is None:
            n_opti_steps = opti_config['steps_in_iter'] * opti_config['n_iters']
        else:
            n_opti_steps = opti_config['early_stop']

        # pbar_optim = tqdm.tqdm(range(n_opti_steps))
        bool_val = True
        
        # to apply truncation trick to w space codes
        w_avg_geo, w_avg_tex = self.model.mapping_geo.w_avg, self.model.mapping_tex.w_avg
                
        # for k in pbar_optim:
        for k in range(n_opti_steps):
            

            use_mse, use_lpips_first_two, use_lpips_last_three = \
                get_optim_state(k, optimizer_all, opti_config, w_avg_geo, w_avg_tex)           
                
            # def render_objects():
            s_render = time.time()
            rgb_out, mask_out, instance_mask, shape_code_i, tex_code_i, depth, lhw, mesh = render_get3D_frame(tracklet,
                                                                                            self.cameras[cam_id],
                                                                                            self.model,
                                                                                            current_fr_id=frame_id,
                                                                                            optimize_w=self.optimize_w,
                                                                                            n_per_step=6,
                                                                                            subsample=self.sample_factor,
                                                                                            device=DEVICE)
            e_render = time.time()

            fg_mask = mask_out > 0
            
            if not fg_mask.sum():
                print("\nNo object rendered inside this image! Increasing sizes\n")
                for obj in list(tracklet.values()):
                    obj[frame_id]['scale'].data = obj[frame_id]['scale'].data/1.15
                # break            

            
            ###################### VISUALIZATION #################################
            # Visualization Draw Bounding Boxes on Image
            if self.no_viz == False:
                
                instances_colored = torch.zeros_like(gt_img)
                grounded_instance_mask = instance_mask.clone() 
                grounded_instance_mask[instance_mask >= len(COLOR20_dict)] = instance_mask[instance_mask >= len(COLOR20_dict)] % len(COLOR20_dict)
                for color_idx, color_RGB in COLOR20_dict.items():
                    instances_colored[grounded_instance_mask == color_idx] = torch.tensor(color_RGB, dtype=torch.float32, device=DEVICE) / 255.
                
                # visualize_depth = depth - depth.min() / -depth.min() 
                full_mask = mask_out[None, None].repeat(1,3,1,1).bool()
                rgb_gt = gt_img.permute(2, 0,1)[None] # imp
                rgb_overlay = rgb_gt.clone() # imp

                rgb_overlay = (rgb_overlay * 0.6) + 1.0 * (1.0 - 0.6)    # fade background
                rgb_overlay[full_mask] = rgb_out.permute(2, 0,1)[None][full_mask].clamp(0.,1.) # adding rgb out overlay to gt
                # dont add rgb out overlay to gt

                if self.no_viz == False:
                    result_cameras = self.cameras[cam_id].clone()
                    rgb_np = np.array(rgb_overlay.permute(0,2,3,1)[0].clamp(0,1).cpu().clone().detach() * 255., dtype=np.uint8)
                    im = Image.fromarray(rgb_np)
                    im_w_bbox = draw_3D_box(im, tracklet, frame_id, result_cameras)
                    
                    # rgb_out bbox
                    rgb_out_np = np.array(rgb_out.permute(2,0,1).clamp(0,1).cpu().clone().detach() * 255., dtype=np.uint8)
                    rgb_out_np = rgb_out_np.transpose(1,2,0)
                    im_rgb_out = Image.fromarray(rgb_out_np)
                    im_w_bbox_rgb_out = draw_3D_box(im_rgb_out, tracklet, frame_id, result_cameras)
                    # rgb_out bbox
                    
                    
                    if k == 0:
                        im_first_setp = im_w_bbox.copy()
                        
                    vis_out = {"rgb_out": rgb_out, # pred rendering
                            "gt_img": gt_img, # gt
                                "instance_mask": instances_colored, # instance mask
                                "im_w_bbox": im_w_bbox, # 3D bbox
                                "im_w_bbox_rgb_out": im_w_bbox_rgb_out, # 3D bbox
                    }  # "depth": visualize_depth, # depth,
                    
                    
                    # only save on the last iteration of the optimization
                    if True: #k == n_opti_steps - 1:
                        vis_dir = os.path.join(save_dir, "vis", f"cam_{str(cam_id)}", f"frame_{str(frame_id)}")
                        os.makedirs(vis_dir, exist_ok=True)
                        for key, val in vis_out.items():
                            if key !=  "im_w_bbox" and key != "im_w_bbox_rgb_out": 
                                val_copy = val.clone().detach().cpu()
                                val_copy = np.array(val_copy.clamp(0,1) * 255., dtype=np.uint8)
                                val_copy = val_copy[0] if key == "depth" else val_copy                            
                                mode = "RGB" if key != "depth" else "L"
                                val_viz = val_copy
                                im = Image.fromarray(val_viz, mode=mode) 
                            else:
                                im = val
                            os.makedirs(os.path.join(vis_dir, f"opt_step_{k}"), exist_ok=True)
                            vis_path = os.path.join(vis_dir, f"opt_step_{k}", f"{key}.png")
                            im.save(vis_path)
                
            ###################### VISUALIZATION #################################
                
            try:
                # RUN TEX only OPTIMIZER with Adam                   
                # def closure():
                    
                if not self.tex_only_optim_adam:
                    optim.zero_grad()
                
                # For lpips get object patches
                gen_patches, gt_patches = get_patch(rgb_out, gt_img, instance_mask.squeeze(0))
                
                # rgb_out, gt_img, fg_mask, gen_patches, gt_patches, instance_mask, shape_code_i, tex_code_i = render_objects()
                s_loss = time.time()
                loss, rgb_loss, lpips_loss, truncation_loss, truncation_loss_geo, truncation_loss_tex = compute_loss(
                    rgb_pred=rgb_out,
                    rgb_gt=gt_img,
                    fg_mask=fg_mask,
                    gen_patches=gen_patches,
                    gt_patches=gt_patches,
                    w_geo=shape_code_i,
                    w_tex=tex_code_i,
                    w_avg_geo=w_avg_geo,
                    w_avg_tex=w_avg_tex,
                    truncation_psi=opti_config["truncation_psi"],
                    truncation_alpha_tex=opti_config["truncation_alpha_tex"],
                    truncation_alpha_geo=opti_config["truncation_alpha_geo"],
                    use_mse=use_mse,
                    use_lpips_first_two=use_lpips_first_two,
                    use_lpips_last_three=use_lpips_last_three,
                    lpips_kwargs=self.lpips_kwargs
                    )
                e_loss = time.time()
                
                
                elapsed_time_s = e_render - s_render
                elipsed_time_loss = e_loss - s_loss
                # loss.backward(retain_graph=True)
                loss.backward()
                elapsed_time_backward = time.time() - e_render
                
                
                
                description = f"{module_name.capitalize()} -" \
                                f"Frame {int(frame_id)} " \
                                f"Camera {int(cam_id)}: " \
                                f"Loss: {float(loss.cpu().detach()):.3f} " \
                                f"- T fwd: {elapsed_time_s:.2f} (s) " \
                                f"- T loss: {elipsed_time_loss:.2f} (s) " \
                                f"- T bck: {elapsed_time_backward:.2f} (s) " \
                                f"- iter: {int(k)}"
                                
                
                for optim in optimizer_all:
                    
                    if not isinstance(optim, list):
                        optim.step()
                    else:
                        for opti_t in optim:
                            opti_t.step()

                for optim in optimizer_all:
                    if not isinstance(optim, list):
                        optim.zero_grad()
                    else:
                        for opti_t in optim:
                            opti_t.zero_grad()
            
            except RuntimeError as e:
                print(e)
                if not fg_mask.sum():
                    print("No object rendered inside the image")
                    continue
                else:
                    print('Can not compute backward or optimize!')
        torch.cuda.empty_cache()
        
        # Update Object Width, Length, Height
        for obj_i, lhw_i, vert_i, face_i in zip(tracklet.values(), lhw, mesh['vert'], mesh['face']):
            final_scale = obj_i[frame_id]['scale']
            lhw_metric= lhw_i / (final_scale+ 1e-6)
            obj_i[frame_id]['length'], obj_i[frame_id]['height'], obj_i[frame_id]['width'] = lhw_metric
            obj_i[frame_id]['mesh'] = {'vert': vert_i, 'face': face_i}

        return tracklet, optimizer_all

    def compute_affinity(self, tracklets, detections):
        embedd_space = 'w' if self.optimize_w else 'z'
        # Init Score
        score = torch.zeros(len(tracklets), len(detections))
        center_distance = torch.zeros(len(tracklets), len(detections))

        # Similatity metrics
        latent_metric = torch.cosine_similarity
        

        if len(tracklets) > 0 and len(detections) > 0:
            track_idx = []
            # latent_weights = torch.tensor([0.2, 0.1, 0.7])[:, None]
            # Compute latent affinity
            detected_latents = {'{}_shape'.format(embedd_space): [], '{}_tex'.format(embedd_space): []}
            for latent_type, latent_ls in detected_latents.items():
                for fitted_detect in detections:
                    latent_ls_to_append = fitted_detect[latent_type]
                    
                    # subtract mean, divide by std
                    if embedd_space == "w":
                        # print(f"old mean and std of latent code {latent_type}: ", latent_ls_to_append.mean(), latent_ls_to_append.std())
                        if latent_type == "w_shape":
                            latent_ls_to_append_normalized = (latent_ls_to_append - self.ws_geo_mean) / self.ws_geo_std
                        elif latent_type == "w_tex":
                            latent_ls_to_append_normalized = (latent_ls_to_append - self.ws_tex_mean) / self.ws_tex_std
                        
                        # print(f"new mean and std of normalized latent code {latent_type}: ", latent_ls_to_append_normalized.mean(), latent_ls_to_append_normalized.std())
                       
                    elif embedd_space == "z":
                        latent_ls_to_append_normalized = latent_ls_to_append # mean and std are already 0 and 1
                    else:
                        raise NotImplementedError(f"Embedding space {embedd_space} not implemented!")
                    
                    latent_ls.append(latent_ls_to_append_normalized.cpu().detach()) 
                    # latent_ls.append(fitted_detect[latent_type].cpu().detach())
 
            detected_latents = {k: torch.stack(v).squeeze() for k, v in detected_latents.items()}

            latent_affinity = []
            for track_id, tracked_obj in tracklets:
                track_idx.append(track_id)
                full_tracklet_latent = []
                full_detect_latent = []
                for key in detected_latents.keys():
                    full_tracklet_latent.append(tracked_obj[key].detach().cpu().squeeze())
                    full_detect_latent.append(detected_latents[key])

                # Get affinity score
                full_latent_score = latent_metric(torch.cat(full_tracklet_latent, dim=-1)[None], torch.cat(full_detect_latent, dim=-1))
                full_latent_score_clamped = full_latent_score.clamp(min=0.)
                latent_affinity.append(full_latent_score_clamped)

            latent_score = torch.stack(latent_affinity)
            score += self.weight_embedding * latent_score

            # Compute position affinity from detections and predict state
            detect_states = [torch.cat([d['state_veh']['translation'],
                                        d['state_veh']['heading'][None],
                                        d['state_veh']['length'][None],
                                        d['state_veh']['width'][None],
                                        d['state_veh']['height'][None]]) for d in detections]
            predict_tracklet_states = [d[1]['predict_state']['state_veh'][:7] for d in tracklets]
            # Get 3D BBOX
            detect_boxes = get_3d_box_corners(torch.stack(detect_states).cpu().detach())
            predict_boxes = get_3d_box_corners(torch.stack(predict_tracklet_states).cpu().detach())
            # Compute IoU
            # for p_box, d_box in zip(predict_boxes, detect_boxes):
            vol, iou = box3d_overlap(predict_boxes, detect_boxes, eps=1e-5)
            score += self.weight_iou * iou

            center_distance = torch.norm(torch.stack(detect_states)[:, :2][None] - torch.stack(predict_tracklet_states)[:, :2][:, None], dim=-1)
            
            # Distance score
            center_dist_score = torch.maximum((-center_distance / self.maximum_distance + 1), torch.zeros(1))
            score += self.weight_center * center_dist_score
            
            
            
            # Distance mask
            center_distance_mask = (center_distance < 15.) # Distances larger 20 m are not considered at all 
            score = score * center_distance_mask

            return score, iou, latent_score, center_distance, center_dist_score

        return None, None, None

    def associate(self, score, center_distance, fitted_detects, avilable_tracklets):
        # Try AB3DMOT Here
        aff_score = np.array(score.T)

        row_ind, col_ind = linear_sum_assignment(-aff_score)  # hougarian algorithm
        matched_indices = np.stack((row_ind, col_ind), axis=1)
        # compute total cost
        cost = 0
        for row_index in range(matched_indices.shape[0]):
            cost -= aff_score[matched_indices[row_index, 0], matched_indices[row_index, 1]]

        # save for unmatched objects
        unmatched_dets = []
        for d in range(len(fitted_detects)):
            if (d not in matched_indices[:, 0]): unmatched_dets.append(d)
        unmatched_trks = []
        for t in range(len(avilable_tracklets)):
            if (t not in matched_indices[:, 1]): unmatched_trks.append(t)

        # filter out matches with low affinity
        matches = []
        for m in matched_indices:
            if (aff_score[m[0], m[1]] < self.affinity_threshold):
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_dets), np.array(unmatched_trks), cost, aff_score

    def write_output(self, tracklets, frame_id, s_pth):
        # Write Output
        output_file_txt = os.path.join(s_pth, 'output.txt')
        with open(output_file_txt, 'a') as f:
            for track_idx in self.active_tracklets:
                track = tracklets[track_idx][frame_id]
                if (track['status'] == 'tracked' or track['status'] == 'lost'):
                    if 'score' in track:
                        score = track['score']
                    else:
                        score = 0.
                    if 'state_veh' in track:
                        veh_state = track['state_veh']

                        f.write('{} {} {} -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                            frame_id, track_idx, 'car', track['alpha'],
                            track['bbox_2D'][0], track['bbox_2D'][1], track['bbox_2D'][2], track['bbox_2D'][3],
                            veh_state['height'], veh_state['width'], veh_state['length'],
                            veh_state['c_x'], veh_state['c_y'], veh_state['c_z'], veh_state['heading'], score))
                    elif 'predict_state' in track:
                        if 'state_veh' in track['predict_state']:
                            veh_state = track['predict_state']['state_veh']

                            f.write('{} {} {} -1 -1 alpha {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                                frame_id, track_idx, 'car', track['alpha'],
                                track['bbox_2D'][0], track['bbox_2D'][1], track['bbox_2D'][2], track['bbox_2D'][3],
                                veh_state[6], veh_state[5], veh_state[4],
                                veh_state[0], veh_state[1], veh_state[2], veh_state[3],
                                score))

    def detection2tracklet(self, detect_3d, camera_pose, step, sdf_scale=1., sdf_shift=torch.tensor([0., 0., 0.]), cam_id=0, cam_name="FRONT", device='cpu'):
        """Take detections/annotations and convert them to tracklets with differentiable features.

        Args:
            detect_3d (_type_): _description_
            camera_pose (_type_): _description_
            step (_type_): _description_
            sdf_scale (_type_, optional): _description_. Defaults to 1..
            sdf_shift (_type_, optional): _description_. Defaults to torch.tensor([0., 0., 0.]).
            cam_id (int, optional): _description_. Defaults to 0.
            device (str, optional): _description_. Defaults to 'cpu'.

        Returns:
            _type_: _description_
        """
        num_obj = len(detect_3d)

        # Init transformations object2cam
        shift_obj, rot_obj, size_obj = \
            get_obj_in_cam(detect_3d, camera_pose, cam_name, device, split=self.split)

        # Init transformations cam2object
        rot_cam = rot_obj.transpose(2, 1)

        # Create tracklet dict
        obj_3d_detect_dict = {str(i + self.detection_count).zfill(5): {'id': None, step: {'rotation': None, 'translation': None, 'scale': None, 'heading': None,
                                                                   'z_shape': None, 'z_map': None, 'z_tex': None,
                                                                   'w_shape': None, 'w_tex': None,
                                                                   'status': None, 'time_lost': 0}} for i in range(num_obj)}
        # Convert rotation to logarithms
        # rot_log = so3_log_map(rot_cam)
        rot_log = matrix_to_axis_angle(rot_cam)
        # Get initial scaling [ -1, 1]
        scale_long_axis = (1 / (torch.max(size_obj, dim=1)[0] / 2))[:, None].to(device)
        # Account for shapenet/get3d -0.5 to 0.5 object scale
        scale_long_axis = scale_long_axis / sdf_scale

        # Init latent codes for each object from code book mean        
        try:
            sdf_code_i = torch.stack([code for code in self.model_codes['sdf'].values()]).mean(dim=0)[None].repeat(num_obj, 1).to(device)
        except:
            sdf_code_i = torch.ones(num_obj, 512, device=DEVICE) * -0.1
        try:
            map_code_i = torch.stack([code for code in self.model_codes['map'].values()]).mean(dim=0)[None].repeat(num_obj, 1).to(device)
        except:
            map_code_i = torch.ones(num_obj, 512, device=DEVICE) * 0.1
        try:
            tex_code_i = torch.stack([code for code in self.model_codes['tex'].values()]).mean(dim=0)[None].repeat(num_obj, 1).to(device)
        except:
            tex_code_i = torch.ones(num_obj, 512, device=DEVICE) * 0.1
            
            
        if self.optimize_w:
            w_shape = self.model.generate_w_geo(sdf_code_i)[:, :1]
            w_tex = self.model.generate_w_tex(tex_code_i)[:, :1]
            apply_truncation_trick(w_shape, self.model.mapping_geo.w_avg, truncation_psi=0.7)
            apply_truncation_trick(w_tex, self.model.mapping_tex.w_avg, truncation_psi=0.7)            

        # Make optimizable values torch parameters
        for i in range(num_obj):
            idx_str = str(i + self.detection_count).zfill(5)
            if not scale_long_axis[i] * sdf_scale > 0.7: # Ignore objects that are too small
                obj_i = obj_3d_detect_dict[idx_str]
                obj_i[step]['heading'] = torch.nn.Parameter(rot_log[None, i, 1].to(device=device), requires_grad=True)
                
                # TODO: Check if center correction works
                shift_obj[i] = shift_obj[i] + sdf_shift.to(shift_obj.device)
                obj_i[step]['translation'] = torch.nn.Parameter(shift_obj[i].to(device=device), requires_grad=True)

                obj_i[step]['scale'] = torch.nn.Parameter(scale_long_axis[i].to(device=device), requires_grad=True)
                
                obj_i[step]['width'] = size_obj[i][0]
                obj_i[step]['length'] = size_obj[i][1]
                obj_i[step]['height'] = size_obj[i][2]

                obj_i[step]['z_shape'] = torch.nn.Parameter(sdf_code_i[i].to(device=device), requires_grad=True)
                obj_i[step]['z_map'] = torch.nn.Parameter(map_code_i[i].to(device=device), requires_grad=True)
                obj_i[step]['z_tex'] = torch.nn.Parameter(tex_code_i[i].to(device=device), requires_grad=True)
                
                if self.optimize_w:
                    obj_i[step]['w_shape'] = torch.nn.Parameter(w_shape[i].to(device=device), requires_grad=True)
                    obj_i[step]['w_tex'] = torch.nn.Parameter(w_tex[i].to(device=device), requires_grad=True)

                obj_i[step]['cam_id'] = cam_id
            else:
                del obj_3d_detect_dict[idx_str]

        self.detection_count = int(len(obj_3d_detect_dict) + self.detection_count)

        return obj_3d_detect_dict

    def tracklet2detection_waymo(self, track_i, camera_pose, frame_id, cam_id=0):
        rot_cam = so3_exp_map(torch.tensor([[0., 1., 0.]], device=DEVICE) * track_i[frame_id]['heading']).detach().cpu()
        rot_obj = rot_cam.transpose(2, 1)

        heading_uncorrected = matrix_to_euler_angles(rot_obj, 'YZX')[0, 0]
        # heading_obj = (eulers[0, 1] + eulers[0, 0])

        heading_correction = np.arctan2(camera_pose.T[0, 1, 0], camera_pose.T[0, 0, 0])

        heading = heading_uncorrected - heading_correction - np.pi
        # Adjust to -pi to pi
        heading_adj = np.mod(heading, 2*np.pi)
        shift_head = heading_adj > np.pi
        heading_adj[shift_head] = heading_adj[shift_head] - 2 * np.pi
        heading_adj = heading_adj.to(torch.float32)

        # Shift
        inv_camera = np.eye(4)
        inv_camera[:3] = np.concatenate(
            [camera_pose[0, :3, :3].T, -np.matmul(camera_pose[0, :3, :3].T, camera_pose[0, :3, -1])[:, None]], axis=1)

        shift_obj = track_i[frame_id]['translation'].detach().cpu()
        if 'center' in track_i[frame_id]:
            shift_obj = shift_obj + track_i[frame_id]['center'].detach().cpu()

        obj_pose_in_cam = shift_obj[[2, 0, 1]]
        obj_xyz = torch.einsum('ij,oj->oi', torch.tensor(inv_camera, dtype=torch.float32),
                               torch.cat([obj_pose_in_cam, torch.ones([1])])[None])[0, :-1]

        w, h, l = track_i[frame_id]['width'].detach().cpu(), track_i[frame_id]['height'].detach().cpu(), track_i[frame_id]['length'].detach().cpu()

        return {'heading': heading_adj, 'translation': obj_xyz, 'width': w,'height': h,'length': l,
                'c_x': obj_xyz[0], 'c_y': obj_xyz[1], 'c_z': obj_xyz[2]}

    def get_global_state(self, state, veh_pose):
        # Get objects from previous step in global frame
        if self.data_set == 'waymo':
            relative_veh_pose = np.einsum('ij,jk->ik', invert_pose(self.start_position), veh_pose)
        else:
            relative_veh_pose = veh_pose
        T = torch.cat([state[:3], torch.tensor([1])])
        
        if self.data_set == 'nuScenes':
            T = T + state[6]/2 * torch.tensor([0., 0., 1., 0.])
            
        T = np.einsum('ij,j->i', relative_veh_pose, np.array(T))[:3]
                
        if self.data_set == 'nuScenes':
            veh_heading_correct = np.arctan2(relative_veh_pose[1,0],relative_veh_pose[0,0])
            heading = state[3] + veh_heading_correct
        else:
            R_obj = np.array(euler_angles_to_matrix(torch.tensor([state[3], 0., 0.]), 'ZYX'))
            R = np.einsum('ij,jm->im', relative_veh_pose[:3, :3], R_obj)
            heading = np.arctan2(R[1, 0], R[0, 0])

        global_state = np.concatenate([T, heading[None], np.array(state)[4:]])
        return global_state

    def get_vehicle_state(self, global_state, veh_pose):
        if self.data_set == 'waymo':
            relative_veh_pose = np.einsum('ij,jk->ik', invert_pose(self.start_position), veh_pose)
        else:
            relative_veh_pose = veh_pose
        inv_relative_veh_pose = invert_pose(relative_veh_pose)
        T = np.concatenate([global_state[:3], np.array([1])])
        R = euler_angles_to_matrix(torch.tensor([0., 0., global_state[3]]), 'XYZ')

        T = np.einsum('ij,j->i', inv_relative_veh_pose, np.array(T))[:3]
        
        if self.data_set == 'nuScenes':
            T = T - global_state[6]/2 * np.array([0., 0., 1.])
        
        if self.data_set == 'nuScenes':
            veh_heading_correct = np.arctan2(relative_veh_pose[1,0],relative_veh_pose[0,0])
            heading = global_state[3] - veh_heading_correct            
        else:
            R = np.einsum('ij,jm->im', inv_relative_veh_pose[:3, :3], np.array(R))
            heading = np.arccos(R[0, 0])

        state = np.array(np.concatenate([T, heading[None], np.array(global_state)[4:]]), dtype=np.float32)
        return torch.tensor(state)

    def get_veh2cam(self, cam2veh):
        veh2cam = np.zeros_like(cam2veh)
        veh2cam[:, -1, -1] = 1.
        veh2cam_rot = cam2veh[:, :3, :3].transpose(0, 2, 1)
        veh2cam_translate = np.einsum('ijk,ik->ij', -veh2cam_rot, cam2veh[:, :3, -1])
        veh2cam[:, :3, :3] = veh2cam_rot
        veh2cam[:, :3, -1] = veh2cam_translate
        return veh2cam