import torch
import torch.nn as nn
import numpy as np
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles
from pytorch3d.transforms import so3_log_map, so3_exp_map
from src.tracking.forward_renderer import _get_object_centric_cameras
import plotly.express as px
from PIL import ImageDraw

from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R


def rotation_from_heading(heading, device='cpu'):
    rot = torch.tensor([[torch.cos(heading), -torch.sin(heading), 0.],
                        [torch.sin(heading), torch.cos(heading), 0.],
                        [0., 0., 1.]], dtype=torch.float32, device=device)
    return rot


def get_obj_in_cam(annotations, camera_pose, cam_name, device='cpu', split='validation'):
    num_obj = len(annotations)
    # Get transformation from vehicle 2 each camera in OpenGL
    if '{}_c_x'.format(cam_name) in annotations[0]:
        prefix = '{}_'.format(cam_name)
    else:
        prefix = ''
    obj_poses = [[obj_val['{}c_x'.format(prefix)], obj_val['{}c_y'.format(prefix)], obj_val['{}c_z'.format(prefix)], 1.]
                 for obj_val in annotations]
    # obj_poses = [[obj_val['c_x'], obj_val['c_y'], obj_val['c_z'], 1.]
    #             for obj_val in annotations]

    obj_pose_in_cam = torch.tensor(np.einsum('ij,oj->oi', camera_pose[0], obj_poses)[:, :3], dtype=torch.float32,
                                   device=device)
    # if not split == 'testing':
    #     obj_pose_in_cam = torch.tensor(np.einsum('ij,oj->oi', camera_pose[0], obj_poses)[:, :3], dtype=torch.float32, device=device)
    # else:
    #     obj_pose_in_cam = torch.tensor(np.stack(obj_poses)[:, :3], dtype=torch.float32, device=device)

    objpose_shift_oGL = obj_pose_in_cam[:, [1, 2, 0]]

    cam_heading_correction = np.arctan2(camera_pose.T[0, 1, 0], camera_pose.T[0, 0, 0])
    # if not split == 'testing':
    #     cam_heading_correction = np.arctan2(camera_pose.T[0, 1, 0], camera_pose.T[0, 0, 0])
    # else:
    #     # TODO: check testing split
    #     cam_from_car = camera_pose[0, :4, :4]
    #     waymo2kitti_RT = np.array([[0, -1, 0, 0], [0, 0, -1, 0],
    #                                [1, 0, 0, 0], [0, 0, 0, 1]])
    #
    #     cam_from_car = np.dot(waymo2kitti_RT, cam_from_car)
    #
    #     car_from_cam = np.linalg.inv(cam_from_car)
    #
    #     rotation_matrix = car_from_cam[:3, :3]
    #     for obj_val in annotations:
    #         rot_y = obj_val['rot_y']
    #         quat = Quaternion(axis=[0, 1, 0], radians=rot_y).elements
    #         x, y, z, w = R.from_matrix(rotation_matrix).as_quat()
    #         w, x, y, z = Quaternion([w, x, y, z]) * quat
    #
    #         heading = R.from_quat([x, y, z, w]).as_euler('xyz')[2]
    #         obj_val['heading'] = heading
    #
    #     cam_heading_correction = np.arctan2(camera_pose.T[0, 1, 0], camera_pose.T[0, 0, 0])

        # cam_heading_correction = np.array(0., dtype=np.float32) + np.pi/2
        # if cam_id == 0: # Front
        #     cam_heading_correction = np.array(0., dtype=np.float32) + np.pi/2
        # elif cam_id == 1: # Front Left
        #     cam_heading_correction = np.array(0., dtype=np.float32)
        # elif cam_id == 2: # Front Right
        #     cam_heading_correction = np.array(0., dtype=np.float32) + np.pi/2
        #     for obj_val in annotations:
        #         obj_val['heading'] = -obj_val['heading']
        # elif cam_id == 3:
        #     cam_heading_correction = np.arctan2(camera_pose.T[0, 1, 0], camera_pose.T[0, 0, 0])
        # elif cam_id == 4:
        #     # cam_heading_correction = np.array(0., dtype=np.float32)
        #     cam_heading_correction = -np.arctan2(camera_pose.T[0, 1, 0], camera_pose.T[0, 0, 0])
        #     # for obj_val in annotations:
        #     #     obj_val['heading'] = -obj_val['heading']
        # else:
        #     cam_heading_correction = np.array(0., dtype=np.float32)

    # if cam_id == 2:
    #     heading_correction = cam_heading_correction + np.pi / 2
    # elif cam_id == 4:
    #     heading_correction = cam_heading_correction - np.pi

    # Correct heading for new camera pose and rotate 90 deg for shapenet models which have x to the left
    ################ TODO: Check for all cameras ################
    corrected_heading = [np.pi + obj_val['{}heading'.format(prefix)] + cam_heading_correction for obj_val in annotations]
    
    # corrected_heading = [obj_val['{}_heading'.format(cam_name)] + cam_heading_correction for obj_val in annotations]
    # # Bound by 0 and 2 * pi
    # corrected_heading_adj = np.mod(np.array(corrected_heading), 2 * np.pi)
    # # Map to [-pi, pi]
    # shift_head = corrected_heading_adj > np.pi
    # corrected_heading_adj[shift_head] = corrected_heading_adj[shift_head] - 2 * np.pi

    # Get openGL rotation from heading angle
    obj_rot_in_cam_oGL = [euler_angles_to_matrix(
        torch.tensor([0., head, 0.], dtype=torch.float32, device=device), 'XYZ')
        for head in corrected_heading]
    obj_rot_in_cam_oGL = torch.stack(obj_rot_in_cam_oGL)

    # Get size of the object
    size_obj = torch.stack(
        [
            torch.tensor([obj['width'], obj['length'], obj['height']], device='cpu', dtype=torch.float32)
            for obj in annotations
        ]
    )

    # write output
    rot_obj = obj_rot_in_cam_oGL
    shift_obj = objpose_shift_oGL
    return shift_obj, rot_obj, size_obj

def waymo_from_obj2cam(shift, rot, scale, inv_scale, size, camera_pose, cam_id):
    annotations = {0: 0}
    return annotations


def get_tracklet_size(tracklet, sdf, sdf_scale, grid_size=20, tolerance=0.005):
    with torch.no_grad():
        xs = torch.linspace(-1, 1, steps=grid_size)
        ys = torch.linspace(-1, 1, steps=grid_size)
        zs = torch.linspace(-1, 1, steps=grid_size)
        x, y, z = torch.meshgrid(xs, ys, zs)
        x_samples = torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)
        x_min = x_samples.clone()
        x_max = x_samples.clone()
        sdf_samples = sdf(x_samples.reshape(1, -1, 3).to('cuda:0'), tracklet['z_shape'][None])
        axis_samples = sdf_samples.reshape(grid_size, grid_size, grid_size)
        surface_mask = (axis_samples < tolerance)
        x_min[~surface_mask] = 1.
        x_max[~surface_mask] = -1.

        min_xyz = torch.min(x_min.reshape(-1, 3), dim=0)[0]
        max_xyz = torch.max(x_max.reshape(-1, 3), dim=0)[0]

        [w, h, l] = torch.abs(max_xyz - min_xyz) * sdf_scale/(tracklet['scale'].detach().cpu())
        center = (min_xyz + max_xyz)/2

        tracklet.update({'width': w, 'height': h, 'length': l, 'center': center})
        return tracklet


def invert_pose(pose):
    pose_inv = np.zeros_like(pose)
    pose_inv[-1, -1] = 1.
    R_inv = pose[:3, :3].T
    t_inv = np.einsum('ij,j->i', R_inv, -pose[:3, -1])

    pose_inv[:3, :3] = R_inv
    pose_inv[:3, -1] = t_inv
    return pose_inv


def get_3d_box_corners(state, eps=1e-2):
    center_x, center_y, center_z, heading, length, width, height = torch.unbind(
        state, dim=1)

    # [N, 3, 3]
    angles = torch.stack([heading, torch.zeros(len(state)), torch.zeros(len(state))]).T
    rotation = euler_angles_to_matrix(angles, 'ZYX')
    # [N, 3]
    # translation = torch.stack([center_x, center_y, center_z], dim=-1)

    # l2 = torch.maximum(length * 0.5, torch.ones_like(length) * eps)
    # w2 = torch.maximum(width * 0.5, torch.ones_like(length) * eps)
    # h2 =  torch.maximum(height * 0.5, torch.ones_like(length) * eps)
    
    xyz = torch.stack([center_x, center_y, center_z], dim=-1)
    whl = torch.stack([torch.maximum(length, torch.ones_like(width) * eps), torch.maximum(width, torch.ones_like(height) * eps), torch.maximum(height, torch.ones_like(length) * eps)], dim=-1)
    
    corners = torch.stack([create_box(p, d) for p, d in zip(xyz, whl)])

    # # [N, 8, 3]
    # corners = torch.reshape(
    #     torch.stack([
    #         l2, w2, -h2, -l2, w2, -h2, -l2, -w2, -h2, l2, -w2, -h2, l2, w2, h2,
    #         -l2, w2, h2, -l2, -w2, h2, l2, -w2, h2
    #     ],
    #         dim=-1), [-1, 8, 3])
    # [N, 8, 3]
    corners = torch.einsum('nij,nkj->nki', rotation, corners)

    return corners


def get_3d_box_corners(state):
    center_x, center_y, center_z, heading, length, width, height = torch.unbind(
        state, dim=1)

    # [N, 3, 3]
    angles = torch.stack([torch.zeros(len(state)), torch.zeros(len(state)), heading]).T
    rotation =  euler_angles_to_matrix(angles, 'ZYX')
    # [N, 3]
    translation = torch.stack([center_x, center_y, center_z], dim=-1)

    l2 = length * 0.5
    w2 = width * 0.5
    h2 = height * 0.5

    # [N, 8, 3]
    corners = torch.reshape(
        torch.stack([
            l2, w2, -h2, -l2, w2, -h2, -l2, -w2, -h2, l2, -w2, -h2, l2, w2, h2,
            -l2, w2, h2, -l2, -w2, h2, l2, -w2, h2
        ],
            dim=-1), [-1, 8, 3])
    # [N, 8, 3]
    corners = torch.einsum('nij,nkj->nki', rotation, corners) + translation[:, None].repeat(1, 8, 1).to(torch.float32)

    return corners

def create_box(xyz, whl):
    x, y, z = xyz
    w, h, le = whl

    verts = torch.tensor(
        [
            [x - w / 2.0, y - h / 2.0, z - le / 2.0],
            [x + w / 2.0, y - h / 2.0, z - le / 2.0],
            [x + w / 2.0, y + h / 2.0, z - le / 2.0],
            [x - w / 2.0, y + h / 2.0, z - le / 2.0],
            [x - w / 2.0, y - h / 2.0, z + le / 2.0],
            [x + w / 2.0, y - h / 2.0, z + le / 2.0],
            [x + w / 2.0, y + h / 2.0, z + le / 2.0],
            [x - w / 2.0, y + h / 2.0, z + le / 2.0],
        ],
        device=xyz.device,
        dtype=torch.float32,
    )
    return verts

def get_3d_box_center(state):
    center_x, center_y, center_z, heading, length, width, height = torch.unbind(
        state, dim=1)

    # [N, 3, 3]
    angles = torch.stack([torch.zeros(len(state)), torch.zeros(len(state)), heading]).T
    rotation = euler_angles_to_matrix(angles, 'XYZ')
    # [N, 3]
    translation = torch.stack([center_x, center_y, center_z], dim=-1)

    l2 = length * 0.
    w2 = width * 0.
    h2 = height * 0.

    # [N, 8, 3]
    corners = torch.reshape(
        torch.stack([
            l2, w2, -h2, -l2, w2, -h2, -l2, -w2, -h2, l2, -w2, -h2, l2, w2, h2,
            -l2, w2, h2, -l2, -w2, h2, l2, -w2, h2
        ],
            dim=-1), [-1, 8, 3])
    # [N, 8, 3]
    corners = torch.einsum('nij,nkj->nki', rotation, corners) + translation[:, None].repeat(1, 8, 1).to(torch.float32)

    return corners[:, :1, :]


def get_2d_box_size_waymo(c_x, c_y, l, w, x_max, y_max):
    l_true = min(x_max, c_x+(l//2)) - max(0, c_x-(l//2))
    w_true = min(y_max, c_y+(w//2)) - max(0, c_y-(w//2))
    return l_true * w_true


def remove_duplicats_in_2D(detections, frame_id, cam_ls, cameras, idx2cam, obj_type, render_factor):
    for i, (c_i, objs_i) in enumerate(detections[frame_id]['projected_lidar'].items()):
        if not len(cam_ls) == i + 1:
            objs_i = {obj_name.split('_' + idx2cam[c_i])[0]: obj_v for obj_name, obj_v in objs_i.items() if
                      obj_type[obj_v['type']] == 'VEHICLE'}

            for c_j in cam_ls[i + 1:]:
                objs_j = detections[frame_id]['projected_lidar'][c_j]
                objs_j = {obj_name.split('_' + idx2cam[c_j])[0]: obj_v for obj_name, obj_v in objs_j.items() if
                          obj_type[obj_v['type']] == 'VEHICLE'}

                for k, o_j in objs_j.items():
                    if k in list(objs_i.keys()):
                        o_i = objs_i[k]
                        img_sz_i = cameras[c_i].image_size[0] * render_factor
                        img_sz_j = cameras[c_j].image_size[0] * render_factor
                        size_i = get_2d_box_size_waymo(o_i['c_x'], o_i['c_y'], o_i['length'], o_i['width'], img_sz_i[1],
                                                       img_sz_i[0])
                        size_j = get_2d_box_size_waymo(o_j['c_x'], o_j['c_y'], o_j['length'], o_j['width'], img_sz_j[1],
                                                       img_sz_j[0])

                        try:
                            if size_j > size_i:
                                del detections[frame_id]['projected_lidar'][c_i][k + '_' + idx2cam[c_i]]
                            else:
                                del detections[frame_id]['projected_lidar'][c_j][k + '_' + idx2cam[c_j]]
                        except:
                            print("Already removed object {}.".format(k))


def get_cam_from_heading_wrt_cam(vehicle_state, camera_poses, FoV, tol=0):
    x_in_camera = torch.einsum('cij,j->ci', torch.tensor(camera_poses, dtype=torch.float32),
                               torch.cat([vehicle_state[: 3], torch.tensor([1.])]))[..., :3]
    cam_angle = torch.rad2deg(torch.atan2(x_in_camera[:, 1], x_in_camera[:, 0]))
    viz_angle_in_cam = FoV/2 - torch.abs(cam_angle)

    if not (torch.sign(viz_angle_in_cam + tol) > 0).any():
        return None

    cam_idx = torch.argmax(viz_angle_in_cam)
    return cam_idx


def draw_3D_box(im, tracklet, frame_id, cameras):
    color_counter = 0
    num_obj = len(tracklet)
    colors = px.colors.qualitative.Light24
    rot_cam = []
    shift_obj = []
    scale = []
    width = []
    height = []
    length = []
    colors = []
    for obj_key, obj_i in tracklet.items():
        if frame_id in obj_i:
            rotation = torch.tensor([[0., 1., 0.]]) * obj_i[frame_id]['heading'].detach().cpu()
            rotation_so3 = so3_exp_map(rotation)
            rot_cam.append(rotation_so3)
            shift_obj.append(obj_i[frame_id]['translation'].detach().cpu())
            scale.append(obj_i[frame_id]['scale'].detach().cpu())
            width.append(obj_i[frame_id]['width'].detach().cpu())
            height.append(obj_i[frame_id]['height'].detach().cpu())
            length.append(obj_i[frame_id]['length'].detach().cpu())
            if 'box_color' in obj_i:
                colors.append(obj_i['box_color'])
                # print("CHECK BOX COLOR: {}", obj_i['box_color'])
            else:
                colors.append(px.colors.qualitative.Light24[np.mod(color_counter, 24)])
                color_counter += 1

    rot_cam = torch.cat(rot_cam, dim=0)
    shift_obj = torch.stack(shift_obj)
    scale = torch.stack(scale)
    width = torch.stack(width)[:, None]
    height = torch.stack(height)[:, None]
    length = torch.stack(length)[:, None]

    whl = scale * torch.cat([width, height, length], dim=-1)
    state = torch.cat([torch.zeros([num_obj, 4]), whl], dim=1)

    for i, track_i in enumerate(tracklet.values()):
        if 'center' in track_i[frame_id]:
            state[i, :3] = track_i[frame_id]['center']

    box_corners = get_3d_box_corners(state)

    obj_cams = _get_object_centric_cameras(R=rot_cam.cpu(), T=shift_obj.cpu(), S=scale.cpu(), base_camera=cameras.cpu()).cpu()

    # TODO: Shift to SDF center
    center_locs = obj_cams.transform_points_screen(box_corners, dtype=np.int32)
    boxes = center_locs[..., :2].to(torch.int32).detach().cpu().numpy()

    draw = ImageDraw.Draw(im)
    for obj_idx, box in enumerate(boxes):
        draw.line(xy=[tuple(box[0]), tuple(box[1])], fill=colors[obj_idx])
        draw.line(xy=[tuple(box[1]), tuple(box[2])], fill=colors[obj_idx])
        draw.line(xy=[tuple(box[2]), tuple(box[3])], fill=colors[obj_idx])
        draw.line(xy=[tuple(box[3]), tuple(box[0])], fill=colors[obj_idx])
        draw.line(xy=[tuple(box[0]), tuple(box[4])], fill=colors[obj_idx])
        draw.line(xy=[tuple(box[1]), tuple(box[5])], fill=colors[obj_idx])
        draw.line(xy=[tuple(box[2]), tuple(box[6])], fill=colors[obj_idx])
        draw.line(xy=[tuple(box[3]), tuple(box[7])], fill=colors[obj_idx])
        draw.line(xy=[tuple(box[4]), tuple(box[5])], fill=colors[obj_idx])
        draw.line(xy=[tuple(box[5]), tuple(box[6])], fill=colors[obj_idx])
        draw.line(xy=[tuple(box[6]), tuple(box[7])], fill=colors[obj_idx])
        draw.line(xy=[tuple(box[7]), tuple(box[4])], fill=colors[obj_idx])

    return im


def get_3D_box(tracklet, frame_id, cameras, center=False):
    state = get_obj_states(tracklet, frame_id)

    if not center:
        # Get corners of the box
        box_points = get_3d_box_corners(state)
    else:
        # Only output the center of the box
        box_points = get_3d_box_center(state)

    rot_cam = []
    shift_obj = []
    scale = []

    for obj_key, obj_i in tracklet.items():
        if frame_id in obj_i:
            rotation = torch.tensor([[0., 1., 0.]]) * obj_i[frame_id]['heading'].detach().cpu()
            rotation_so3 = so3_exp_map(rotation)
            rot_cam.append(rotation_so3)
            shift_obj.append(obj_i[frame_id]['translation'].detach().cpu())
            scale.append(obj_i[frame_id]['scale'].detach().cpu())

    rot_cam = torch.cat(rot_cam, dim=0)
    shift_obj = torch.stack(shift_obj)
    scale = torch.stack(scale)

    obj_cams = _get_object_centric_cameras(R=rot_cam.cpu(), T=shift_obj.cpu(), S=scale.cpu(),
                                           base_camera=cameras.cpu()).cpu()

    center_locs = obj_cams.transform_points_screen(box_points)
    boxes = center_locs[..., :2].to(torch.float32).detach().cpu().numpy()

    return boxes


def get_obj_states(tracklet, frame_id):
    num_obj = len(tracklet)

    scale = []
    width = []
    height = []
    length = []
    for obj_key, obj_i in tracklet.items():
        if frame_id in obj_i:
            scale.append(obj_i[frame_id]['scale'].detach().cpu())
            width.append(obj_i[frame_id]['width'].detach().cpu())
            height.append(obj_i[frame_id]['height'].detach().cpu())
            length.append(obj_i[frame_id]['length'].detach().cpu())

    scale = torch.stack(scale)
    width = torch.stack(width)[:, None]
    height = torch.stack(height)[:, None]
    length = torch.stack(length)[:, None]

    whl = scale * torch.cat([width, height, length], dim=-1)
    state = torch.cat([torch.zeros([num_obj, 4]), whl], dim=1)

    for i, track_i in enumerate(tracklet.values()):
        if 'center' in track_i[frame_id]:
            state[i, :3] = track_i[frame_id]['center']

    return state


def kitti2vehicle(annotations, camera_pose):

    cam_from_car = camera_pose[0, :4, :4]
    waymo2kitti_RT = np.array([[0, -1, 0, 0], [0, 0, -1, 0],
                               [1, 0, 0, 0], [0, 0, 0, 1]])

    cam_from_car = np.dot(waymo2kitti_RT, cam_from_car)

    car_from_cam = np.linalg.inv(cam_from_car)

    rotation_matrix = car_from_cam[:3, :3]
    for obj_val in annotations:
        rot_y = obj_val['rot_y']
        quat = Quaternion(axis=[0, 1, 0], radians=rot_y).elements
        x, y, z, w = R.from_matrix(rotation_matrix).as_quat()
        w, x, y, z = Quaternion([w, x, y, z]) * quat

        heading = R.from_quat([x, y, z, w]).as_euler('xyz')[2]
        obj_val['heading'] = heading

        x, y, z = obj_val['translation']
        translation = np.matmul(car_from_cam,
                                np.array([x, y, z, 1]).T).tolist()[:3]

        obj_val['c_x'] = translation[0]
        obj_val['c_y'] = translation[1]
        obj_val['c_z'] = translation[2]

    return annotations


# def rot_y2alpha(rot_y, x, FOCAL_LENGTH):
#     """
#     Get alpha by rotation_y - theta
#     rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
#     x : Object center x to the camera center (x-W/2), in pixels
#     alpha : Observation angle of object, ranging [-pi..pi]
#     """
#     alpha = rot_y - np.arctan2(x, FOCAL_LENGTH)
#     alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
#     return alpha
#
#
# def waymo2kitti(tracklets, camera_pose):
#     instance_id_dict[obj.id +
#                      lidar] = len(instance_id_dict)
#
#
#     x1, y1, x2, y2 = id_to_bbox.get(obj.id + lidar)
#     x = tracklets['c_x']
#     y = tracklets['c_y']
#     z = tracklets['c_z']
#     h = tracklets['height']
#     w = tracklets['width']
#     l = tracklets['length']
#     rot_y = tracklets['heading']
#
#     transform_box_to_cam = cam_from_car @ get_box_transformation_matrix(
#         (x, y, z), (l, h, w), rot_y)
#     pt1 = np.array([-0.5, 0.5, 0, 1.])
#     pt2 = np.array([0.5, 0.5, 0, 1.])
#     pt1 = np.matmul(transform_box_to_cam, pt1).tolist()
#     pt2 = np.matmul(transform_box_to_cam, pt2).tolist()
#
#     new_loc = np.matmul(cam_from_car,
#                         np.array([x, y, z,
#                                   1]).T).tolist()
#     x, y, z = new_loc[:3]
#     rot_y = -math.atan2(pt2[2] - pt1[2],
#                         pt2[0] - pt1[0])
#
#
#     rot_y = -math.atan2(pt2[2] - pt1[2],
#                         pt2[0] - pt1[0])
#     alpha = rot_y2alpha(rot_y, x, z).item()
#
#     alpha = alpha,
#     roty = rot_y,
#     dimension = [float(dim) for dim in [h, w, l]],
#     translation = [float(loc) for loc in [x, y, z]],