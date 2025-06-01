import os
import json
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

idx2cam = {0: 'FRONT', 1: 'FRONT_LEFT', 2: 'FRONT_RIGHT', 3: 'SIDE_LEFT', 4: 'SIDE_RIGHT'}
cam_name2cam = {'FRONT': dataset_pb2.CameraName.FRONT,
                'FRONT_LEFT': dataset_pb2.CameraName.FRONT_LEFT,
                'FRONT_RIGHT': dataset_pb2.CameraName.FRONT_RIGHT,
                'SIDE_LEFT': dataset_pb2.CameraName.SIDE_LEFT,
                'SIDE_RIGHT': dataset_pb2.CameraName.SIDE_RIGHT}

def _create_object(tracklet_at_fi, tracklet_idx, timestap, context_name):
    """Creates a prediction objects file."""

    o = metrics_pb2.Object()
    o_dict = {}
    # The following 3 fields are used to uniquely identify a frame a prediction
    # is predicted at. Make sure you set them to values exactly the same as what
    # we provided in the raw data. Otherwise your prediction is considered as a
    # false negative.
    o.context_name = context_name
    o_dict['context_name'] = context_name
    # The frame timestamp for the prediction. See Frame::timestamp_micros in
    # dataset.proto.
    invalid_ts = -1
    o.frame_timestamp_micros = timestap
    o_dict['frame_timestamp_micros'] = timestap
    # This is only needed for 2D detection or tracking tasks.
    # Set it to the camera name the prediction is for.
    cam_name = idx2cam[int(tracklet_at_fi['cam_id'])]
    o.camera_name = cam_name2cam[cam_name]
    o_dict['camera_name'] = cam_name2cam[cam_name]

    # Populating box and score.
    box = label_pb2.Label.Box()
    box.center_x = float(tracklet_at_fi['state_veh']['translation'][0].detach().cpu())
    box.center_y = float(tracklet_at_fi['state_veh']['translation'][1].detach().cpu())
    box.center_z = float(tracklet_at_fi['state_veh']['translation'][2].detach().cpu())
    box.length = float(tracklet_at_fi['state_veh']['length'].detach().cpu())
    box.width = float(tracklet_at_fi['state_veh']['width'].detach().cpu())
    box.height = float(tracklet_at_fi['state_veh']['height'].detach().cpu())
    box.heading = float(tracklet_at_fi['state_veh']['heading'].detach().cpu())
    o.object.box.CopyFrom(box)
    o_dict['center_x'] = float(tracklet_at_fi['state_veh']['translation'][0].detach().cpu())
    o_dict['center_y'] = float(tracklet_at_fi['state_veh']['translation'][1].detach().cpu())
    o_dict['center_z'] = float(tracklet_at_fi['state_veh']['translation'][2].detach().cpu())
    o_dict['height'] = float(tracklet_at_fi['state_veh']['height'].detach().cpu())
    o_dict['length'] = float(tracklet_at_fi['state_veh']['length'].detach().cpu())
    o_dict['width'] = float(tracklet_at_fi['state_veh']['width'].detach().cpu())
    o_dict['heading'] = float(tracklet_at_fi['state_veh']['heading'].detach().cpu())
    # This must be within [0.0, 1.0]. It is better to filter those boxes with
    # small scores to speed up metrics computation.
    o.score = float(tracklet_at_fi['score'])
    o_dict['score'] = float(tracklet_at_fi['score'])
    # For tracking, this must be set and it must be unique for each tracked
    # sequence.
    o.object.id = str(tracklet_idx)
    # o.object.id = tracklet_idx
    o_dict['id'] = str(tracklet_idx)
    # Use correct type.
    o.object.type = label_pb2.Label.TYPE_VEHICLE
    o_dict['type'] = label_pb2.Label.TYPE_VEHICLE

    o_dict['bbox2D_left'] = float(tracklet_at_fi['bbox_2D'][0])
    o_dict['bbox2D_top'] = float(tracklet_at_fi['bbox_2D'][1])
    o_dict['bbox2D_right'] = float(tracklet_at_fi['bbox_2D'][1])
    o_dict['bbox2D_bottom'] = float(tracklet_at_fi['bbox_2D'][3])
    o_dict['alpha'] = float(tracklet_at_fi['alpha'])
    o_dict['status'] = str(tracklet_at_fi['status'])

    # Add more objects. Note that a reasonable detector should limit its maximum
    # number of boxes predicted per frame. A reasonable value is around 400. A
    # huge number of boxes can slow down metrics computation.
    return o, o_dict

def write_scene_waymo(tracklets, scene, out_pth):
    objects = metrics_pb2.Objects()
    tracklets_out = {}
    for track_idx, tracklet_i in tracklets.items():
        track_i_dict = {}
        for k, v in tracklet_i.items():
            if isinstance(k, int) and (v['status'] == 'tracked' or v['status'] == 'lost'):
                frame_id = k
                frame_i = scene[frame_id]
                o, o_dict = _create_object(v, k, timestap=frame_i['timestamp_micro'], context_name=frame_i['context_name'])
                objects.objects.append(o)
                track_i_dict[frame_id] = o_dict
        tracklets_out[track_idx] = track_i_dict

    # Write objects to a file.
    out_file_bin = os.path.join(out_pth, 'output.bin')
    f = open(out_file_bin, 'wb')
    f.write(objects.SerializeToString())
    f.close()

    out_file_json = os.path.join(out_pth, 'output.json')
    with open(out_file_json, "w") as outfile:
        json.dump(tracklets_out, outfile)
