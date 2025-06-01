import os
import json
from pytorch3d.transforms import matrix_to_quaternion, euler_angles_to_matrix
import torch
import numpy as np

def write_scene_nuscenes(tracklets, scene, out_pth=None):
    
    results = {} # {sample_token <str>: List[sample_result]} for given scene
    scene_name = scene.name
    scene_number = scene_name.split("-")[1]
    sample_tokens = scene.sample_tokens
    
    for all_sample_token in sample_tokens:
        results[all_sample_token] = []
    
    # index of each sample_token in scene is the keyframe_idx
    keyframe_idx2token = {i: sample_token for i, sample_token in enumerate(sample_tokens)}
   
    print("Writing results for scene-{}...".format(scene_number))
        
    for _, tracklet in tracklets.items():
        global_object_id = scene_number + str(tracklet["id"]).zfill(6)  # unique object id across all scenes
                
        tracked_frames_idx = [k for k, v in tracklet.items() if isinstance(k, int) and v['status'] != 'dead']
        tracked_status = [v['status'] for k, v in tracklet.items() if isinstance(k, int) and v['status'] != 'dead']
        for stat in reversed(tracked_status):
            if stat == 'lost':
                tracked_frames_idx.pop(-1)
            if stat == 'tracked':
                break
                
        
        for k in tracked_frames_idx:
            v = tracklet[k]
            if isinstance(k, int) and v['status'] != 'dead':
                keyframe_idx = k
                tracking_score = v["score"] # original tracking score - positive
                    
                global_state = v["global_state"]
                try:
                    tracking_score = v["score"]
                except:
                    tracking_score = 0.1
                translation = global_state[:3]
                heading = global_state[3:4] # aka heading
                # Heading to rotation quaternion
                rotation = np.array(matrix_to_quaternion(euler_angles_to_matrix(torch.tensor([torch.tensor([heading]), 0., 0.]), 'ZYX')), dtype=np.float32)
                length = global_state[4:5]
                width = global_state[5:6]
                height = global_state[6:7]
                velocity = global_state[7:9] * 2 # convert from m per frame to m/s at 2Hz assuming 2 FPS keyframes
                    
                
                sample_token = keyframe_idx2token[keyframe_idx]
                    
                sample_result = {
                    "sample_token": str(sample_token),                      # <str> Foreign key. Identifies the sample/keyframe for which objects are detected.,
                    "translation": [float(x) for x in translation],   # <float> [3]   -- Estimated bounding box location in meters in the global frame: center_x, center_y, center_z.,
                    "size": [float(width), float(length), float(height)], # <float> [3]   -- Estimated bounding box size in meters: width, length, height.,
                    "rotation": [float(x) for x in rotation],        # <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.,
                    "velocity": [float(x) for x in velocity],          # <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.,
                    "tracking_id": str(global_object_id),                   # <str>         -- Unique object id that is used to identify an object track across samples.,
                    "tracking_name": str("car"),                            # <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.,
                    "tracking_score": float(tracking_score)                    # <float> [0, 1] -- Object prediction score between 0 and 1 for the class identified by tracking_name.
                }
    
                
                results[sample_token].append(sample_result)
    
    if out_pth is not None:
        os.makedirs(os.path.join(out_pth, "detection_results"), exist_ok=True)
        out_file_json = os.path.join(out_pth, "detection_results/scene-{}.json".format(scene_number))
        with open(out_file_json, "w") as f:
            json.dump(results, f, indent=4)
            print("Wrote results to {}".format(out_file_json))
          
    return results
