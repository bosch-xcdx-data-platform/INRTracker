import torch
import torch.nn as nn

ALL_STATES = ['translation', 'rotation', 'scale', 'sdf', 'map', 'texture']
ALL_CAMERAS_WAYMO = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_RIGHT', 'SIDE_LEFT']
cam2idx = {'FRONT': 0, 'FRONT_LEFT': 1, 'FRONT_RIGHT': 2, 'SIDE_RIGHT': 3, 'SIDE_LEFT': 4}


def optimize_waymo_tracklets(tracklets, forward_renderer, intrinsic, extrinsic, img_w, img_h, sphere_tracer, model,
                             code_book, images, projected_detections, resizing_factor=1.,
                             camera_names=ALL_CAMERAS_WAYMO,  optim_states=ALL_STATES):
    # init optimizers
    for state in optim_states:
        # TODO: Stuff
        torch.optim.Adam(state)

    # Iterate through all cameras
    for cam in camera_names:
        idx = cam2idx[cam]
        # Run only on objects visible in the frame
        visible_tracklets = projected_detections[idx]

        # Render frame for visible tracklets
        rgb_pred, fg_mask, code_book = forward_renderer(visible_tracklets, intrinsic, extrinsic, img_w, img_h,
                                                        sphere_tracer, model, code_book, resizing_factor)
        rgb_loss = nn.L1Loss(rgb_pred[fg_mask], images[fg_mask])
        latent_loss = 0.
        volume_loss = 0.
        # TODO: Figure out a way to build potential loss on kalman predictions (Start without)
        pose_potential_loss = 0.

    # Update tracklets

    return tracklets