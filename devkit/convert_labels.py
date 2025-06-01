import numpy as np
import os 
import json
import pickle as pkl
import argparse
from tqdm import tqdm
from collections import defaultdict
import copy

# def read_json(tracking_result_path):
#     with open(tracking_result_path, 'r') as f:
#         tracking_result = json.load(f)
#
#     return tracking_result




class Cameras(object):
    id_camera_dict = {
        'front': 1,
        'front_left': 2,
        'front_right': 3,
        'side_left': 4,
        'side_right': 5
    }
    def __init__(self, camera_names=None):
        if camera_names is None:
            self.camera_names = self.id_camera_dict.keys()
        else:
            self.camera_names = camera_names
        
        self.cameras = defaultdict()

    def get_camera(self, camera_name):
        pass

class WaymoDetectionLoader(object):
    def __init__(self, root, prefix=None, exp_name=None, precompiled=False, camera_names=None, split='testing', store_filename='WaymoDetectionLoader_storage.json', store_total=True, store_per_recording=True, chunk_max=16, recording=None):
        self.cameras = Cameras(camera_names=camera_names)
        self.storage = defaultdict(self.set_default_dict) 
        self.split = split
        self.root = root
        self.store_filename = store_filename
        self.store_per_recording = store_per_recording
        self.store_total = store_total
        self.chunk_max = chunk_max
        if precompiled == False:
            #Reload data from scratch and build lookup table
            self.output_jsons = self.get_paths(prefix, exp_name)
            print(self.output_jsons)
            # output_test_pure_det_3dcen
            # prefix quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter_scaled_res_16
            self.raw_datas = [self.read_json(os.path.join(root, i)) for i in self.output_jsons]
            for raw_data in self.raw_datas:
                self.iterate_json_tree(raw_data)
            if self.store_per_recording or self.store_total:
                self.dump_storage()
        else:
            path_stored = os.path.join(self.root, self.store_filename)
            if recording is None:
                if os.path.exists(path_stored):
                    self.storage = self.read_json(path_stored)
                else:
                    assert False, 'stored file does not exist {}'.format(path_stored)
            else:
                file_path = os.path.join(self.root, self.store_filename.split('.json')[0]+recording+'.json')
                if os.path.exists(file_path):
                    self.storage = self.read_json(file_path)
                else:
                    assert False, 'stored file does not exist {}'.format(file_path)



    def get_paths(self, prefix, experiment_name):
        """
        Filename is now hardcoded to be named output-json.
        """

        output_jsons = list()
        if prefix == None:
            output_jsons = [i for i in os.listdir(self.root) if i.endswith('.json')]
        else:
            # if there is a prefix ouputs are located in subdirectories named output.json
            # iterate over chunks
            for i in range(self.chunk_max):
                if experiment_name is None:
                    folder = os.path.join(self.root, prefix+'_'+str(i), 'output.json')
                else:
                    folder = os.path.join(self.root, prefix+'_'+str(i), experiment_name, 'output.json')
                if os.path.exists(folder):
                    output_jsons.append(folder)
        return output_jsons

    def dump_storage(self):
        if self.store_total:
            with open(os.path.join(self.root, self.store_filename),'w') as f:
                #f.writelines(json.dumps(self.storage, indent=4))
                json.dump(self.storage, f)
            print('Dumped: {}'.format(os.path.join(self.root, self.store_filename)))
        if self.store_per_recording:
            for recording in self.storage:
                print(recording)
            for recording in self.storage:
                with open(os.path.join(self.root, self.store_filename.split('.json')[0]+'_'+recording+'.json'),'w') as f:
                    #f.writelines(json.dumps({recording: self.storage[recording]}))       #, indent=4
                    json.dump({recording: self.storage[recording]}, f)
                print('Dumped: {}'.format(os.path.join(self.root, self.store_filename.split('.json')[0]+'_'+recording+'.json')))
    

    def set_default_dict(self):
        cameras_dict = dict()
        for camera in self.cameras.camera_names:
            cameras_dict[camera] = defaultdict(list)
        return cameras_dict

    def push_back_sample(self, recording, split, camera, image_frame, annotation):
        assert self.split == split, 'trying to push back a label from a different split {}/{}'.format(split, self.split)
        self.storage[recording][camera][image_frame].append(annotation)

    def iterate_json_tree(self, raw_data):
        all_image_idx = dict()
        
        for idx, image in enumerate(tqdm(raw_data['images'])):
            camera, split, recording, frame_name = image['file_name'].split('/')[-4:]
            all_image_idx[idx] = (camera, split, recording, frame_name)

        for idx, label in enumerate(tqdm(raw_data['annotations'])):
            image_id = label['image_id']
            camera, split, recording, frame_name = all_image_idx[image_id]
            # append image to right frame
            image_frame = frame_name.split('_')[-1].split('.png')[0]
            self.push_back_sample(recording, split, camera, image_frame, copy.deepcopy(label))

    def get_sample(self, recording, camera, image_frame):
        # do robust getting with none to identify missing
        label = self.storage[recording][camera][image_frame]
        return label

    @staticmethod
    def read_json(path):
        with open(path, 'r') as f:
            tracking_result = json.load(f)
        return tracking_result

    def return_anno(self, recording, camera, image_frame, min_score=0.3, max_depth=75):
        annos = self.get_sample(recording, camera, image_frame)
        out = []
        saved_ids = []
        for anno in annos:
            tmp = dict()
            c_x, c_y, c_z = anno['translation']
            w, h, l = anno['dimension']
            roty = anno['roty']
            cat_id = anno['category_id']
            depth = anno['depth']
            score = anno['score']
            if cat_id == 3 and depth[0]<max_depth and score>min_score and not anno['id'] in saved_ids:
                tmp['c_x'] = c_x
                tmp['c_y'] = c_y
                tmp['c_z'] = c_z
                tmp['translation'] = anno['translation']
                tmp['c_x_man'] = c_z
                tmp['c_y_man'] = -c_x
                tmp['c_z_man'] = -c_y
                tmp['width'] = w
                tmp['height'] = h
                tmp['length'] = l
                tmp['rot_y'] = roty
                tmp['score'] = score
                tmp['depth'] = depth
                out.append(tmp)
                saved_ids.append(anno['id'])
        return out


class nuScenesDetectionLoader(object):
    def __init__(self) -> None:
        pass

if __name__ == '__main__':
    # root = '/home/julian/workspace/qd-3dt/work_dirs/Waymo/test'
    root = '/nas/EOS/users/jost/results_qd-3dt/work_dirs/Waymo'
    prefix = 'quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter_scaled_res_16'
    exp_name = 'output_test_pure_det_3dcen'

    store_filename = 'qd3dt_waymo_detections_stored.json'

    # tracking_result_path = os.path.join(root, prefix + '_0', exp_name, 'output.json')
    # data_dict = read_json(tracking_result_path=tracking_result_path)

    #tracking_result_path = os.path.join('data/file_16_0', 'output.json')
    # data_dict = read_json(tracking_result_path=tracking_result_path)
    # wd = WaymoDetectionLoader('/home/mb4257/Repos/qd-3dt/scripts/data/', prefix='file_16')
    wd = WaymoDetectionLoader(root=root, prefix=prefix, exp_name=exp_name, store_filename=store_filename, store_total=False)
    # print(data_dict.keys())
