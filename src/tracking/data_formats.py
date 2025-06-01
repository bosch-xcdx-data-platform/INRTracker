import json
from typing import List
import torch
import tensorflow as tf
if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()
import os
import numpy as np
from waymo_open_dataset import dataset_pb2 as open_dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from torch.utils.data import Dataset
import cv2
from PIL import Image
import src.tracking.tracking_utils as tu
import pickle as pkl

WAYMO_CAM_INTEGER_2_NAME = {0: 1, 1: 2, 2: 4, 3: 3, 4: 5}
WAYMO_CAM_NAME_2_INTEGER = {1: 0, 2: 1, 4: 2, 3: 3, 5: 4}

def get_rotation(roll, pitch, heading):
    s_heading = np.sin(heading)
    c_heading = np.cos(heading)
    rot_z = np.array([[c_heading, -s_heading, 0], [s_heading, c_heading, 0], [0, 0, 1]])

    s_pitch = np.sin(pitch)
    c_pitch = np.cos(pitch)
    rot_y = np.array([[c_pitch, 0, s_pitch], [0, 1, 0], [-s_pitch, 0, c_pitch]])

    s_roll = np.sin(roll)
    c_roll = np.cos(roll)
    rot_x = np.array([[1, 0, 0], [0, c_roll, -s_roll], [0, s_roll, c_roll]])

    rot = np.matmul(rot_z, np.matmul(rot_y, rot_x))

    return rot

def invert_transformation(rot, t):
    t = np.matmul(-rot.T, t)
    inv_translation = np.concatenate([rot.T, t[:, None]], axis=1)
    return np.concatenate([inv_translation, np.array([[0.0, 0.0, 0.0, 1.0]])])

cats_mapping = {
    'pedestrian': 1,
    'cyclist': 2,
    'car': 3,
    'truck': 4,
    'tram': 5,
    'misc': 6,
    'dontcare': 7
}
cats_kitti2waymo = {
    1 : 2,
    2 : 4,
    3 : 1,
    4 : 1,
    5 : 0,
    6 : 0,
    7 : 0
}
types_waymo = {0: 'UNKNOWN', 1: 'VEHICLE', 2: 'PEDESTRIAN', 3: 'SIGN', 4: 'CYCLIST'}
kitti_cats = {
    'Pedestrian': 'pedestrian',
    'Cyclist': 'cyclist',
    'cyclist': 'cyclist',
    'car': 'car',
    'Car': 'car',
    'Van': 'car',
    'Truck': 'truck',
    'Tram': 'tram',
    'Person': 'pedestrian',
    'pedestrian': 'pedestrian',
    'Person_sitting': 'pedestrian',
    'Misc': 'misc',
    'DontCare': 'dontcare'
}

def invA(A):
    isBatch = (A.dim() == 3)
    if not isBatch:
        A = A[None]

    R = A[..., :3, :3]
    t = A[:, :3, 3]
    R_T = R.transpose(-2,-1)
    t_inv = torch.einsum('bnm, bm -> bn', R_T, -t)
    A_inv = torch.eye(4)[None].repeat((len(A), 1, 1))
    A_inv[..., :3, :3] = R_T
    A_inv[..., :3, 3] = t_inv

    if not isBatch:
        A_inv = A_inv.squeeze()
    
    return A_inv

class KITTITracking(Dataset):
    
    def __init__(self, base_dir, split='testing', label_root=None, format='tracking'):
        assert split in ['training', 'testing'] #validation
        assert format in ['tracking', 'detection'], 'tracking and detection only allowed in format'
        self.data_dir = os.path.join(base_dir, split, 'image_02')
        self.base_dir = base_dir
        self.split = split
        self.scenes = [path for path in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir,path))]
        self.name = None

        if label_root is None:
            if format!='tracking':
                print('Error:::: Per default the labelroot loads tacking data. Check format and labelroot for KITTITracking dataloader.')
                exit(-1)

        self.label_root = label_root
        self.format = format
        self.cam2idlabel = {0: "FRONT"}
        self.label2camid = {"FRONT": 0}
    
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):
        scene_name = sorted(self.scenes)[idx]
        calib_path = os.path.join(self.base_dir, self.split, 'calib', scene_name + '.txt')
        images_path = os.path.join(self.base_dir, self.split, 'image_02', scene_name)
        oxts_path = os.path.join(self.base_dir, self.split, 'oxts', scene_name + '.txt')

        if self.label_root is None:
            label_path = os.path.join(self.base_dir, self.split, 'label_02', scene_name + '.txt')
        elif self.format == 'tracking':
            label_path = os.path.join(self.label_root, scene_name + '.txt')
        elif self.format == 'detection':
            label_path = os.path.join(self.label_root, scene_name)
        else:
            print('Why are you here! Name format either tracking or detection in KITTI dataloader!')

        frames = [os.path.join(images_path, i) for i in sorted(os.listdir(images_path)) if i.endswith('.png')] 
        
        scene = KITTITrackingScene(frames, label_path, calib_path, oxts_path, self.format, split=self.split)
        scene.name = scene_name
        return scene
class KITTITrackingScene(Dataset):
    def __init__(self, frames, label_path, calib_path, oxts_path, format, split):
        self.intrinsics = np.array([self.read_calib(calib_path)]) # write in list with one element because only one camera is used
        self.oxts = self.oxts_to_pose(self.read_oxts(oxts_path))

        veh2cam = [self.read_calib(calib_path, cam=6)]
        self.veh2cam = np.eye(4)[None].repeat(len(self.intrinsics), 0)
        self.veh2cam[:, :3, :4] = np.array(veh2cam)

        self.cam2veh = np.eye(4)[None].repeat(len(self.intrinsics), 0)
        self.cam2veh[:, :3, :4] = np.array([self.invert_calibration(i) for i in veh2cam])

        self.velo2cam = [self.read_calib(calib_path, cam=5)]
        self.cam2velo = [self.invert_calibration(i) for i in self.velo2cam]
        self.frames = frames
        self.format = format
        self.label_path = label_path
        self.all_labels = None
        self.name = None

        self.split = split

        if format == 'tracking':
            # @TODO discuss is image_id really importatn
            self.all_labels = self.load_tracking(self.label_path, adjust_center=True)


    def load_detection(self, detections_path, adjust_center=True):
        all_detections = self.load_oxts(detections_path)
        annotations = list()
        for detection in all_detections:
            kitti_properties = detection.split()

            x1, y1, x2, y2 = float(kitti_properties[4]), float(kitti_properties[5]), float(
                                kitti_properties[6]), float(kitti_properties[7])

            if adjust_center:
                # KITTI GT uses the bottom of the car as center (x, 0, z).
                # Prediction uses center of the bbox as center (x, y, z).
                # So we align them to the bottom center as GT does
                y_cen_adjust = float(kitti_properties[8]) / 2.0
            else:
                y_cen_adjust = 0.0
            center_2d = tu.cameratoimage(
                    np.array([[
                        float(kitti_properties[11]),
                        float(kitti_properties[12])- y_cen_adjust,
                        float(kitti_properties[13])
                    ]]), self.intrinsics[0]).flatten().tolist()


            object_dict = {
                    'category_id':cats_mapping[kitti_cats[ kitti_properties[0]]],
                    'identity': kitti_properties[0],
                    'truncated': float(kitti_properties[1]),
                    'occlusion': float(kitti_properties[2]),
                    'angle': float(kitti_properties[3]),
                    'xleft': int(round(float(kitti_properties[4]))),
                    'ytop': int(round(float(kitti_properties[5]))),
                    'xright': int(round(float(kitti_properties[6]))),
                    'ybottom': int(round(float(kitti_properties[7]))),
                    'height': float(kitti_properties[8]),
                    'width': float(kitti_properties[9]),
                    'length': float(kitti_properties[10]),
                    'dimension': [
                               float(kitti_properties[8]),
                               float(kitti_properties[9]),
                               float(kitti_properties[10])
                           ],
                    'posx': float(kitti_properties[11]),
                    'posy': float(kitti_properties[12]),
                    'posz': float(kitti_properties[13]),
                    'translation':[
                               float(kitti_properties[11]),
                               float(kitti_properties[12])- y_cen_adjust,
                               float(kitti_properties[13])
                           ],
                    'roty': float(kitti_properties[14]),
                    'bbox':[x1, y1, x2 - x1, y2 - y1],
                    'area':(x2 - x1) * (y2 - y1),
                    'center_2d':center_2d,
                    'delta_2d':[
                               center_2d[0] - (x1 + x2) / 2.0,
                               center_2d[1] - (y1 + y2) / 2.0
                           ],
                }
            if len(object_dict)>17:
                object_dict['score'] =  float(kitti_properties[18])

            annotations.append(object_dict)
        return annotations

    def load_tracking(self, tracking_path, adjust_center=False):
        self.labels_by_frame_cam = {}
        self.labels_by_frame = {}
        trackid_maps = dict()
        annotations = list()
        global_track_id = 0
        ann_id = 0
        labels = self.read_labels(tracking_path)
        for label in labels:
                image_id = label[0]
                cat = label[2]
                if cat in ['DontCare', 'Cyclist', 'Pedestrian']:
                    continue
                image_id = image_id
                if label[1] in trackid_maps.keys():
                    track_id = trackid_maps[label[1]]
                else:
                    track_id = global_track_id
                    trackid_maps[label[1]] = track_id
                    global_track_id += 1
                x1, y1, x2, y2 = float(label[6]), float(label[7]), float(
                    label[8]), float(label[9])

                if adjust_center:
                    # KITTI GT uses the bottom of the car as center (x, 0, z).
                    # Prediction uses center of the bbox as center (x, y, z).
                    # So we align them to the bottom center as GT does
                    y_cen_adjust = float(label[10]) / 2.0
                else:
                    y_cen_adjust = 0.0

                center_2d = tu.cameratoimage(
                    np.array([[
                        float(label[13]),
                        float(label[14]) - y_cen_adjust,
                        float(label[15])
                    ]]), self.intrinsics[0]).flatten().tolist()

                ann = dict(id=ann_id,
                           image_id=image_id,
                           category_id=cats_mapping[kitti_cats[cat]],
                           type=cats_kitti2waymo[cats_mapping[kitti_cats[cat]]],
                           instance_id=track_id,
                           alpha=float(label[5]),
                           rot_y=float(label[16]),
                           dimension=[
                               float(label[10]),
                               float(label[11]),
                               float(label[12])
                           ],
                           height=float(label[10]),
                           width=float(label[11]),
                           length=float(label[12]),
                           translation=[
                               float(label[13]),
                               float(label[14]) - y_cen_adjust,
                               float(label[15])
                           ],
                           c_x=float(label[15]),
                           c_y=-float(label[13]),
                           c_z=-float(label[14]) - y_cen_adjust,
                           is_occluded=int(label[4]),
                           is_truncated=float(label[3]),
                           center_2d=center_2d,
                           delta_2d=[
                               center_2d[0] - (x1 + x2) / 2.0,
                               center_2d[1] - (y1 + y2) / 2.0
                           ],
                           bbox=[x1, y1, x2 - x1, y2 - y1],
                           area=(x2 - x1) * (y2 - y1),
                           iscrowd=False,
                           ignore=False,
                           segmentation=[[x1, y1, x1, y2, x2, y2, x2, y1]])

                if not int(image_id) in self.labels_by_frame_cam:
                    self.labels_by_frame_cam[int(image_id)] = {}
                    self.labels_by_frame[int(image_id)] = {}

                self.labels_by_frame_cam[int(image_id)].update({str(ann_id).zfill(5) + '_FRONT_LEFT': ann})
                self.labels_by_frame[int(image_id)].update({str(ann_id).zfill(5): ann})
                annotations.append(ann)
                ann_id += 1
        return annotations

    @staticmethod
    def invert_calibration(calib):
        R = np.linalg.inv(calib[:3,:3])
        T = -np.matmul(R, calib[:,-1])
        out = np.hstack((R,T[:,np.newaxis]))
        return out

    @staticmethod
    def read_calib(calib_path, cam=2):
        """ Read calibration file and return camera matrix
            e.g.,
                projection = read_calib(cali_dir, vid_id)
        """
        with open(calib_path) as f:
            fields = [line.split() for line in f]
        return np.asarray(fields[cam][1:], dtype=np.float32).reshape(3, 4)

    @staticmethod
    def read_oxts(oxts_path):
        """ Read oxts file and return each fields for KittiPoseParser
            e.g., 
                fields = read_oxts(oxt_dir, vid_id)
                poses = [KittiPoseParser(field) for field in fields]
        """
        with open(oxts_path, 'r') as f:
            fields = [[float(split) for split in line.strip().split()] for line in f]
        return fields


    @staticmethod
    def read_labels(labels_path):
        """ Read oxts file and return each fields for KittiPoseParser
            e.g.,
                fields = read_oxts(oxt_dir, vid_id)
                poses = [KittiPoseParser(field) for field in fields]
        """
        with open(labels_path, 'r') as f:
            fields = [line.strip().split() for line in f]
        return fields

    def oxts_to_pose(self, oxts):
        poses = []

        def latlon_to_mercator(lat, lon, s):
            r = 6378137.0
            x = s * r * ((np.pi * lon) / 180)
            y = s * r * np.log(np.tan((np.pi * (90 + lat)) / 360))
            return [x, y]

        lat0 = oxts[0][0]
        scale = np.cos(lat0 * np.pi / 180)
        pose_0_inv = None
        for oxts_val in oxts:
            pose_i = np.zeros([4, 4])
            pose_i[3, 3] = 1

            [x, y] = latlon_to_mercator(oxts_val[0], oxts_val[1], scale)
            z = oxts_val[2]
            translation = np.array([x, y, z])

            roll = oxts_val[3]
            pitch = oxts_val[4]
            heading = oxts_val[5]
            rotation = get_rotation(roll, pitch, heading)

            pose_i[:3, :] = np.concatenate([rotation, translation[:, None]], axis=1)
            if pose_0_inv is None:
                pose_0_inv = invert_transformation(pose_i[:3, :3], pose_i[:3, 3])

            pose_i = np.matmul(pose_0_inv, pose_i)
            poses.append(pose_i)

        return np.array(poses)

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, item):
        frame = {}

        image = cv2.imread(self.frames[item])
        image = np.array(Image.open(self.frames[item]))
        n_cameras = 1

        camera_calibrations = self.intrinsics


        frame['height'] = np.array(image.shape[0])[None]
        frame['width'] = np.array(image.shape[1])[None]

        frame['cam2veh'] = self.cam2veh
        frame['veh2cam'] = self.veh2cam
        frame['intrinsics'] = np.array([self.intrinsics[:, 0,0], self.intrinsics[:, 1,1], self.intrinsics[:, 0,2],  self.intrinsics[:,1,2], self.intrinsics[:,0,3], self.intrinsics[:,1,3]]).T

        frame['veh_pose'] = self.oxts[item]
        
        if self.format == 'detection':
            labels = self.load_detection(os.path.join(self.label_path, self.frames[item].split('/')[-1].split('.')[0]))
        elif self.format == 'tracking':
            lidar_labels = self.labels_by_frame[item]
            camera_labels = self.labels_by_frame_cam[item]
        else:
            print(f'What are you doing here! You should choose tracking or detection for format. {self.format}')
        
        # Get Object Labels
        # Only one camera 0 available
        frame['lidar_labels'] = lidar_labels
        frame['camera_labels'] = {0: camera_labels}

        # Get all Images
        frame['imgs'] = [image]

        return frame

class WaymoTune(Dataset):
    def __init__(self, base_dir, split='validation'):
        assert split in ['training', 'validation', 'testing']
        self.data_dir = os.path.join(base_dir, split)
        self.split = split
        self.scenes = [path for path in os.listdir(self.data_dir) if path[-9:] == '.tfrecord']

        self.camid2label = {0: "FRONT", 1: "FRONT_LEFT", 2: "FRONT_RIGHT", 3: "SIDE_LEFT", 4:"SIDE_RIGHT"}
        self.label2camid = {"FRONT": 0, "FRONT_LEFT": 1, "FRONT_RIGHT": 2, "SIDE_LEFT": 3, "SIDE_RIGHT": 4}

        self.n_frames = len(self)
        self.frames_by_scene = [list(range(len(self.__getitem__(i)))) for i in range(len(self))]
        
        # Waymo uses tfrecord -->  Restrict memory usage

    def __len__(self):
        return np.sum([len(self.__getitem__(i)) for i in range(len(self))])

    def __getitem__(self, idx):
        scene_name = self.scenes[idx]
        print("Loading Scene {}".format(scene_name))
        scene_pth = os.path.join(self.data_dir, scene_name)
        scene_file = tf.data.TFRecordDataset(scene_pth, compression_type="")
        scene = WaymoTrackingScene(scene_file)
        scene.name = scene_name[:-9]
        return scene

class WaymoTracking(Dataset):
    def __init__(self, base_dir, split='validation'):
        assert split in ['training', 'validation', 'testing']
        self.data_dir = os.path.join(base_dir, split)
        self.split = split
        self.scenes = [path for path in os.listdir(self.data_dir) if path[-9:] == '.tfrecord']

        self.camid2label = {0: "FRONT", 1: "FRONT_LEFT", 2: "FRONT_RIGHT", 3: "SIDE_LEFT", 4:"SIDE_RIGHT"}
        self.label2camid = {"FRONT": 0, "FRONT_LEFT": 1, "FRONT_RIGHT": 2, "SIDE_LEFT": 3, "SIDE_RIGHT": 4}

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):            
        scene_name = self.scenes[idx]
        print("Loading Scene {}".format(scene_name))
        scene_pth = os.path.join(self.data_dir, scene_name)
        self._limit_tf_gpu_usage()
        scene_file = tf.data.TFRecordDataset(scene_pth, compression_type="")
        scene = WaymoTrackingScene(scene_file)
        scene.name = scene_name[:-9]
        return scene
    
    def _limit_tf_gpu_usage(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)

class WaymoTrackingScene(Dataset):
    def __init__(self, scene_file):
        self.intrinsics = None
        self.cam2veh = None
        self.img_sz = None
        self.frames = {}
        self.name = None
        for f_num, data in enumerate(scene_file):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            self.frames.update({f_num: frame})

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):
        frame = {}
        frame_compressed = self.frames[item]
        n_cameras = len(frame_compressed.images)

        camera_calibrations = [frame_compressed.context.camera_calibrations[i] for i in range(n_cameras)]
        frame['context_name'] = frame_compressed.context.name
        frame['timestamp_micro'] = frame_compressed.timestamp_micros

        # Extract Intrinsics
        if self.intrinsics is not None:
            frame['intrinsics'] = self.intrinsics
        else:
            intrinsics = np.stack(
                [np.array(camera_calibrations[i].intrinsic, dtype=np.float32)
                 for i in range(n_cameras)]
            )
            self.intrinsics = intrinsics
            frame['intrinsics'] = intrinsics

        if self.img_sz is not None:
            frame['height'] = self.img_sz[0]
            frame['width'] = self.img_sz[1]
        else:
            h = np.stack(
                [np.array(camera_calibrations[i].height, dtype=np.float32)
                 for i in range(n_cameras)]
            )
            w = np.stack(
                [np.array(camera_calibrations[i].width, dtype=np.float32)
                 for i in range(n_cameras)]
            )
            self.img_sz = np.empty((2, n_cameras))
            self.img_sz[0] = h
            self.img_sz[1] = w
            frame['height'] = h
            frame['width'] = w

        # Extract Extrinsics
        if self.cam2veh is not None:
            frame['cam2veh'] = self.cam2veh
        else:
            extrinsics = np.stack(
                [np.array(camera_calibrations[i].extrinsic.transform, dtype=np.float32).reshape(4, 4)
                 for i in range(n_cameras)]
            )
            self.cam2veh = extrinsics
            frame['cam2veh'] = extrinsics

        # Get Vehcile pose
        frame['per_cam_veh_pose'] = np.stack(
            [np.array(frame_compressed.images[WAYMO_CAM_NAME_2_INTEGER[i + 1]].pose.transform, dtype=np.float32).reshape(4,4)
             for i in range(n_cameras)]
        )
        frame['veh_pose'] = np.array(frame_compressed.pose.transform, dtype=np.float32).reshape(4,4)

        # Get Object Labels
        frame['lidar_labels'] = {label.id: self._extract_label_fields(label, 3) for label in frame_compressed.laser_labels}
        frame['camera_labels'] = {i: {obj_label.id:
                                          self._extract_label_fields(obj_label, 2)
                                      for obj_label in proj_li_label.labels}
                                  for i, proj_li_label in enumerate(frame_compressed.projected_lidar_labels)
                                  if len(proj_li_label.labels) > 0}

        # Get all Images
        frame['imgs'] = [tf.image.decode_jpeg(frame_compressed.images[WAYMO_CAM_NAME_2_INTEGER[i + 1]].image).numpy()
                         for i in range(n_cameras)]

        return frame

    def _extract_label_fields(self, l, dims):
        assert dims in [2, 3]
        label_dict = {
            "c_x": l.box.center_x,
            "c_y": l.box.center_y,
            "width": l.box.width,
            "length": l.box.length,
            "type": l.type,
        }
        if dims == 3:
            label_dict["c_z"] = l.box.center_z
            label_dict["height"] = l.box.height
            label_dict["heading"] = l.box.heading
            most_visivle_in = l.most_visible_camera_name
            label_dict["visible_cam"] = most_visivle_in
            label_dict["{}_c_x".format(most_visivle_in)] = l.camera_synced_box.center_x
            label_dict["{}_c_y".format(most_visivle_in)] = l.camera_synced_box.center_y
            label_dict["{}_c_z".format(most_visivle_in)] = l.camera_synced_box.center_z
            label_dict["{}_heading".format(most_visivle_in)] = l.camera_synced_box.heading
        return label_dict

class nuScenesTracking(Dataset):
    def __init__(self, 
                 dataroot="data/nuscenes", 
                 detection_root="data/centertrack_detections", 
                 split='testing', 
                 verbose=False,
                 start_idx=0,
                 stop_idx=-1,
                 threshold=0.20 # threshold for detection score, default 0.20
                 ):
        assert split in ['training', 'validation', 'testing', 'mini']
        if split == 'testing':
            version = 'v1.0-test'
        elif split == 'mini':
            version = 'v1.0-mini'
        else:
            version = 'v1.0-trainval'
            
        self.dataroot = dataroot
        self.version = version
        self.split = split
        self.threshold = threshold
        
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
        
        print("Loaded nuScenes version {} with {} samples and {} scenes from the {} split".format(version, len(self.nusc.sample), len(self.nusc.scene), split))
        
        self.scene_names = [scene['name'] for scene in self.nusc.scene]
        
        if stop_idx == -1:
            stop_idx = len(self.scene_names) # if stop_idx is not specified, use all scenes
        
        self.scene_names = self.scene_names[start_idx:stop_idx] # split scenes into smaller chunks
        
        # # val split
        # self.scene_names = ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
        #                     'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
        #                     'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
        #                     'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
        #                     'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
        #                     'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
        #                     'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
        #                     'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
        #                     'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626',
        #                     'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
        #                     'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780',
        #                     'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797',
        #                     'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
        #                     'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
        #                     'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
        #                     'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
        #                     'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059',
        #                     'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067',
        #                     'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073']
        
        
        # # save scene names as pkl file
        # with open(os.path.join("data", "ablations", "nuscenes", "scene_names_{}_{}_{}.pkl".format(start_idx, stop_idx, version)), 'wb') as f:
        #     pkl.dump(self.scene_names, f)
        # print("Saved scene names to {}".format(os.path.join("data", "ablations", "nuscenes", "scene_names_{}_{}_{}.pkl".format(start_idx, stop_idx, version))))
        
        print("Loading scenes numbers {} to {}".format(start_idx, stop_idx))
        print("first scene name: {}".format(self.scene_names[0]))
        print("last scene name: {}".format(self.scene_names[-1]))
        if version == 'v1.0-test':
            detection_namespace = "_test"
        else:
            detection_namespace = "_val"
        
        self.detection_file_path = os.path.join(detection_root, "results{}.json".format(detection_namespace))
        print("Loading detections from {}.".format(self.detection_file_path))
        self.detection_results, self.detection_meta = self._load_detections(self.detection_file_path)
        
        # mapping between camera IDs and labels in nuScenes
        self.camid2label = {0: "FRONT", 1: "FRONT_LEFT", 2: "FRONT_RIGHT", 3: "BACK_LEFT", 4: "BACK_RIGHT", 5: "BACK"}
        self.label2camid = {"FRONT": 0, "FRONT_LEFT": 1, "FRONT_RIGHT": 2, "BACK_LEFT": 3, "BACK_RIGHT": 4, "BACK": 5}
        
        # legacy: from  WaymoTracking class
        self.data_dir = self.dataroot
        self.scenes = self.scene_names
        
    def _load_detections(self, detection_file_path):
        with open(detection_file_path, 'r') as f:
            detection_dict = json.load(f)
        detection_results, detection_meta = detection_dict['results'], detection_dict['meta']

        return detection_results, detection_meta 
             
    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, scene_idx):            
        scene_name = self.scene_names[scene_idx]
        print("Loading Scene {}".format(scene_name))
        tracking_scene = nuScenesTrackingScene(self.nusc, 
                                                scene_name, 
                                                self.detection_results,
                                                self.threshold
                                                ) # Get tracking scene
        return tracking_scene
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.dataroot}, {self.split})"
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.dataroot}, {self.split})"
    
# nuScenesTrackingScene class
class nuScenesTrackingScene(Dataset):
    def __init__(self, 
                 nusc: NuScenes, # NuScenes object
                 scene_name: str, # scene name
                 detection_results: dict,
                 threshold: float, # threshold for detection score
                 ): # detection results
        
        self.nusc = nusc
        self.scene_name = scene_name # scene name, e.g. 'scene-0001'
        self.scene_token = self.nusc.field2token('scene', 'name', scene_name)[0] # scene token
        self.scene = self.nusc.get('scene', self.scene_token) # scene 
        
        self.detection_results = detection_results # dict of detection_results
        
        self.threshold = threshold # threshold for detection score
        
        self.sample_tokens = self.nusc.field2token('sample', 'scene_token', self.scene_token) # samples, aka keyframes, in this scene
        self.samples = [self.nusc.get('sample', sample_token) for sample_token in self.sample_tokens] # get samples
        
        # All sensor readings (lidar, camera, radar) are stored in the sample_data table. 
        # Some of these are keyframes as indicated by is_key_frame, others are not.
        # If you start from the sample table then you get only keyframes, 
        # but you can use the next and prev pointers to move forward/backward in time, 
        # which also includes non-keyframes.
        
        # self.first_sample_token = self.scene['first_sample_token'] # first sample token
        # self.last_sample_token = self.scene['last_sample_token'] # last sample token
        
        # self.first_sample = self.nusc.get('sample', self.first_sample_token) # first sample
        # self.last_sample = self.nusc.get('sample', self.last_sample_token) # last sample
        
        # self.first_sample_data_token = self.first_sample['data']['CAM_FRONT']
        # self.last_sample_data_token = self.last_sample['data']['CAM_FRONT']
        
        # current_sample_data_token = self.first_sample_data_token # current sample data token
        # self.sample_tokens = [] # list of sample tokens
        # self.samples = [] # list of samples
        # self.sample_data_tokens = [] # list of sample data tokens
        # self.sample_data = [] # list of sample data
        
        # def add_sample_data(current_sample_data_token):
        #     current_sample_data = self.nusc.get('sample_data', current_sample_data_token)
        #     current_sample = self.nusc.get('sample', current_sample_data['sample_token'])
        #     self.sample_tokens.append(current_sample['token'])
        #     self.samples.append(current_sample)
        #     self.sample_data_tokens.append(current_sample_data['token'])
        #     self.sample_data.append(current_sample_data)
        #     return current_sample_data['next']
        
        # while current_sample_data_token != self.last_sample_data_token:
        #     current_sample_data_token = add_sample_data(current_sample_data_token)
        
        # add_sample_data(current_sample_data_token) # add last sample data
                 
        # legacy - from waymoTrackingScene class
        self.frames = self.samples
        self.name = self.scene_name
        
        self.intrinsics = None
        self.cam2veh = None
        self.img_sz = None
        
        self.camid2label = {0: "FRONT", 1: "FRONT_LEFT", 2: "FRONT_RIGHT", 3: "BACK_LEFT", 4: "BACK_RIGHT", 5: "BACK"}
        self.label2camid = {"FRONT": 0, "FRONT_LEFT": 1, "FRONT_RIGHT": 2, "BACK_LEFT": 3, "BACK_RIGHT": 4, "BACK": 5}

        self.sensor_detection_id2label = {1: 'FRONT', 2: 'FRONT_RIGHT', 3: 'BACK_RIGHT', 4:'BACK',5: 'BACK_LEFT', 6: 'FRONT_LEFT'}
        self.label2sensor_detection_id = {'FRONT': 1, 'FRONT_RIGHT': 2, 'BACK_RIGHT': 3, 'BACK': 4, 'BACK_LEFT': 5, 'FRONT_LEFT': 6}
        
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, keyframe_idx):
        keyframe = {} # frame in waymoTrackingScene
        
        sample = self.nusc.get('sample', self.sample_tokens[keyframe_idx]) # get keyframe, aka sample   
        
        # get sample data tokens
        sample_data_tokens = list(sample['data'].values()) 
        # get sample data
        sample_data_list = [self.nusc.get('sample_data', sample_data_token) for sample_data_token in sample_data_tokens] 
        # get all calibrated sensors
        calibrated_sensors = [self.nusc.get('calibrated_sensor', sample_data["calibrated_sensor_token"]) for sample_data in sample_data_list] 
        # Get mapping to relevant sensors
        sensor_names = [self.nusc.get('sensor', calibrated_sensor.get('sensor_token'))['channel'] for calibrated_sensor in calibrated_sensors]
        sensor_nu_mapping = [[l, k] for l, sensor in enumerate(self.label2camid.keys()) for k, nu_sensor in enumerate(sensor_names) if "CAM_" + sensor == nu_sensor]
        nu2label_idx = torch.tensor(sensor_nu_mapping)[:, 1]
        
        n_cameras = len(sensor_nu_mapping)
        
        # legacy: waymoTrackingScene 
        frame_compressed = sample
        camera_calibrations = calibrated_sensors 
        # n_cameras = n_sensors
        
        keyframe['context_name'] = self.scene_name
        keyframe['timestamp_micro'] = sample['timestamp']
        
        # Extract Intrinsics
        if self.intrinsics is not None:
            keyframe['intrinsics'] = self.intrinsics
        else:
            intrinsics = torch.stack([torch.tensor(calibrated_sensors.get('camera_intrinsic')) if len(calibrated_sensors.get('camera_intrinsic')) else torch.eye(3) for calibrated_sensors in camera_calibrations ])[nu2label_idx]
            focal_x, focal_y, c_u, c_v = intrinsics[:, 0,0], intrinsics[:, 1,1], intrinsics[:, 0,2], intrinsics[:,1,2]
            self.intrinsics = torch.stack([focal_x, focal_y, c_u, c_v]).T.numpy()
            keyframe['intrinsics'] = torch.stack([focal_x, focal_y, c_u, c_v]).T.numpy()
            

        if self.img_sz is not None:
            keyframe['height'] = self.img_sz[0]
            keyframe['width'] = self.img_sz[1]
        else:   
            h = torch.tensor([sample_data["height"] for sample_data in sample_data_list])[nu2label_idx].numpy()
            w = torch.tensor([sample_data["width"] for sample_data in sample_data_list])[nu2label_idx].numpy()
            
            self.img_sz = np.empty((2, n_cameras))
            self.img_sz[0] = h
            self.img_sz[1] = w
            keyframe['height'] = h
            keyframe['width'] = w

        # Extract Extrinsics
        # All extrinsic parameters are given with respect to the ego vehicle body frame, as calibrated on a particular vehicle
        translations = torch.tensor([calib_sensor.get('translation') for calib_sensor in camera_calibrations])[nu2label_idx].numpy()
        rotations = torch.tensor([Quaternion(calib_sensor.get('rotation')).rotation_matrix for calib_sensor in camera_calibrations])[nu2label_idx].numpy()
        
        if self.cam2veh is not None:
            keyframe['cam2veh'] = self.cam2veh
        else:
            # Rotation and translation of the sensor relative to the ego vehicle body frame
            extrinsics = torch.eye(4)[None].repeat(len(translations),1,1).numpy()
            extrinsics[:, :3, :3] =  np.matmul(rotations, np.array([[0., -1., 0.],[0., 0., -1.],[1., 0., 0.]]))
            extrinsics[:, :3, 3] = translations
            self.cam2veh = extrinsics
            # cam2veh are the camera extrinsics wrt to the vehicle
            keyframe['cam2veh'] = extrinsics
        
        # Get Vehcile pose
        ego_pose_tokens = [sample_data["ego_pose_token"] for sample_data in sample_data_list] # get ego pose tokens
        ego_pose = [self.nusc.get('ego_pose', ego_pose_token) for ego_pose_token in ego_pose_tokens] # get ego pose
        # Ego vehicle pose: given with respect to global coordinate system of the log's map
        ego_pose_translations = torch.tensor([pose.get('translation') for pose in ego_pose])[nu2label_idx].numpy()
        ego_pose_rotations = torch.tensor([Quaternion(pose.get('rotation')).rotation_matrix for pose in ego_pose])[nu2label_idx].numpy()
        
        # veh_pose is the pose of the vehicle in world coordinates
        veh_pose = torch.eye(4)[None].repeat(len(ego_pose_translations),1,1).numpy()
        veh_pose[:, :3, :3] = ego_pose_rotations
        veh_pose[:, :3, 3] = ego_pose_translations
        keyframe['veh_pose'] = veh_pose[0]
        
        # world2veh = invA(torch.tensor(veh_pose)).numpy()
        
        # Get all Images
        filenames = [sample_data["filename"] for sample_data in sample_data_list]
        # exclude non-camera paths
        filenames = [filenames[idx[1]] for idx in sensor_nu_mapping]
        # read images as numpy arrays
        keyframe['imgs'] = [np.array(Image.open(os.path.join(self.nusc.dataroot, filename))) for filename in filenames]
    
        # per_cam_veh_pose is the synchronized version with each camera. 
        # Given that we are using camera detections, this should be the the same as veh_pose
        keyframe['per_cam_veh_pose'] = veh_pose
        def assign_id(detections):
            detections_w_id = {}
            for idx, det in enumerate(detections):
                det_id = str(idx).zfill(5)
                detections_w_id[det_id] = det
            return detections_w_id

        # detection_results: keys = sample_token, values = list of detection dict
        detections_in_sample = self.detection_results.get(sample['token']) # get detection results in sample
        # filter detections by threshold score
        if detections_in_sample is None:
            detections_in_sample = []
        else:
            detections_in_sample = [detection for detection in detections_in_sample if detection.get('detection_score') > self.threshold]
        # assign unique id to each detection
        detections_w_id = assign_id(detections_in_sample) # assign unique id to each detection
        
        # camera_labels are the 2D detections per camera. 
        # This is a nested dictionary with the camera ids as keys and list of detection instance tokens as values
        keyframe["camera_labels"] = {}
        keyframe['lidar_labels'] = {} 

        for det_id, det in detections_w_id.items():
            sensor_detection_name = self.sensor_detection_id2label[det.get("sensor_id")]
            det_i_sensor_id = self.label2camid[sensor_detection_name]
            
            # veh_per_cam2veh = torch.matmul(invA(torch.tensor(keyframe['veh_pose'])), torch.tensor(veh_pose[det_i_sensor_id]))
            
            # world2veh = invA(torch.tensor(veh_pose[det_i_sensor_id])).numpy()
            world2veh = invA(torch.tensor(veh_pose[0])).numpy()
            veh_heading_correct = np.arctan2(veh_pose[det_i_sensor_id][1,0], veh_pose[det_i_sensor_id][0,0])
            
            det_translation = np.array(det.get("translation")+[1])
            det_translation_veh = np.matmul(world2veh, det_translation) # Map to vehicle space
            
            q = Quaternion(det.get("rotation"))
            v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
            yaw = np.arctan2(v[1], v[0]) - veh_heading_correct # Map to vehicle pose
            # yaw = np.array(Quaternion(det.get("rotation")).yaw_pitch_roll)[0]
            det_type = det.get("detection_name")
            if det_type == 'car':
                det_type = 1
                if sensor_detection_name == "FRONT":
                    a = 0
            else:
                det_type = 0
            nested_dict = {
                "c_x": det_translation_veh[0],
                "c_y": det_translation_veh[1],
                "c_z": det_translation_veh[2] - det.get("size")[2] / 2.,
                "width": det.get("size")[0],
                "length": det.get("size")[1],
                "height": det.get("size")[2],
                "heading": yaw, # yaw_pitch_roll
                "score": det.get("detection_score"),   
                "type": det_type,
                "sensor_id": det_i_sensor_id,
            }
            keyframe['lidar_labels'][f"{det_id}"] = nested_dict
            
            if not det_i_sensor_id in keyframe["camera_labels"]:
                keyframe['camera_labels'][det_i_sensor_id] = {}
                
            keyframe['camera_labels'][det_i_sensor_id][f"{det_id}_{sensor_detection_name}"] = {"type": det_type}

        return keyframe
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.scene_name})"
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.scene_name})"
