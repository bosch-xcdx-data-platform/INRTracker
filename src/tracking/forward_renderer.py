import os
import pickle
import time

import torch
import torch.nn as nn
import functools
import multiprocessing as mp
import numpy as np
from PIL import Image

from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix
from pytorch3d.structures.meshes import join_meshes_as_scene
from src.datasets.shapenet_core_custom import ShapeNetCoreCustom
from pytorch3d.renderer import \
    MultinomialRaysampler, \
    MeshRenderer, \
    MeshRasterizer, \
    HardFlatShader, \
    RasterizationSettings

from pytorch3d.transforms import so3_exp_map
from torchvision.transforms.functional import resize as torch_resize
from torchvision.transforms.functional import crop as torch_crop
from torchvision.transforms.functional import _interpolation_modes_from_int
from torchvision.transforms.functional import resized_crop as torch_resized_crop


from src.datasets.shapenet_core_custom import mesh_from_verts_face_tex

# GET3D submodule
# from uni_rep.camera.perspective_camera import PerspectiveCamera as PerspectiveCamera_GET3D
from submodules.GET3D.uni_rep.camera.perspective_camera import PerspectiveCamera as PerspectiveCamera_GET3D

torch.autograd.set_detect_anomaly(True)


def _get_mesh_params(data_set, ids):
    obj_synset_id, obj_model_id = ids
    model_path = os.path.join(
        data_set.shapenet_dir, obj_synset_id, obj_model_id, data_set.model_dir
    )
    verts, faces, textures = data_set._load_mesh(model_path)
    return {"verts": verts, "faces": faces, "textures": textures}


def get_mesh_on_demand(batch, data_set):
    num_meshes = len(batch['synset_id'])

    G = functools.partial(_get_mesh_params, data_set)
    print("Loading {} not renderd meshes!".format(num_meshes))

    pool = mp.Pool(np.minimum(mp.cpu_count(), num_meshes))
    results = pool.map(G, [(obj_synset_id, obj_model_id) for obj_synset_id, obj_model_id in zip(batch["synset_id"], batch["model_id"])])

    batch.update({k: [mesh_params[k] for mesh_params in results] for k in results[0].keys()})

    return batch


def run_mesh_rendering_pipeline(camera, mesh, device, raster_settings):
    assert len(camera) == 1
    assert len(mesh) == 1
    mesh_rasterizer = MeshRasterizer(cameras=camera.to(device),
                                     raster_settings=raster_settings)
    lights = None
    shader = HardFlatShader(
        device=device,
        cameras=camera.to(device),
        lights=lights)
    renderer = MeshRenderer(
        rasterizer=mesh_rasterizer,
        shader=shader,
    )

    rgb_obj = renderer(mesh)
    return rgb_obj


def render_object_centric_mesh_scene(single_cam, data_points, rot_cam, shift_obj, raster_settings, device):
    # Render objects with one camera per object
    meshes = mesh_from_verts_face_tex(data_points["verts"],
                                      data_points["faces"],
                                      data_points["textures"]
                                      if "textures" in data_points
                                      else None).to(device)
    num_obj = len(meshes)

    # Init object centered cameras
    object_cams = PerspectiveCameras(
        focal_length=single_cam.focal_length.repeat(num_obj, 1),
        principal_point=single_cam.get_principal_point().repeat(num_obj, 1),
        R=rot_cam,
        T=shift_obj,
        image_size=single_cam.get_image_size(),
        in_ndc=False)

    rgb_all_objs = []
    cam_obj_distance = []
    for i in range(len(meshes)):
        rgb_obj_i = run_mesh_rendering_pipeline(object_cams[i], meshes[i], device, raster_settings)
        rgb_all_objs.append(rgb_obj_i)
        cam_obj_distance.append(torch.norm(object_cams[i].T))

    rgb_merged = rgb_all_objs[np.argsort(cam_obj_distance)[0]]
    for place_i in np.argsort(cam_obj_distance)[1:]:
        non_occluded_mask = (rgb_merged[..., -1] == 0.)
        rgb_merged[non_occluded_mask] = rgb_all_objs[place_i][non_occluded_mask]

    rgb_img_np = np.array(rgb_merged.cpu().detach() * 255., np.uint8)
    im = Image.fromarray(rgb_img_np.squeeze())
    im.save('./.tmp/joined_mesh_waymo_centered_obj.png')

    return rgb_merged


def render_camera_centric_mesh_scene(single_cam, data_points, trafo_obj, raster_settings, device):
    # Render objects at different locations and a single camera
    # Transform object
    data_points["verts"] = [
        torch.matmul(trafo_obj[i], torch.cat([verts, torch.ones_like(verts)[:, :1]], dim=-1).T)[:3].T
        for i, verts in enumerate(data_points["verts"])
    ]

    meshes = mesh_from_verts_face_tex(data_points["verts"],
                                      data_points["faces"],
                                      data_points["textures"]
                                      if "textures" in data_points
                                      else None).to(device)
    meshes = join_meshes_as_scene(meshes)

    rgb_img_i = run_mesh_rendering_pipeline(single_cam, meshes, device, raster_settings)

    rgb_img_np = np.array(rgb_img_i.cpu().detach() * 255., np.uint8)
    im = Image.fromarray(rgb_img_np.squeeze())
    im.save('./.tmp/joined_mesh_waymo_centered_cam.png')

    # img_h = single_cam.image_size[:, 0]
    # img_w = single_cam.image_size[:, 1]
    #
    # raysampler = MultinomialRaysampler(min_x=-1.0, max_x=1.0, min_y=-1.0, max_y=1.0,
    #                                    image_width=img_w, image_height=img_h, n_pts_per_ray=0,
    #                                    min_depth=0.1, max_depth=1.0, unit_directions=True, )
    # rays = raysampler(single_cam.to(device))
    #
    # import open3d as o3d
    # n_plt_pts = 4
    # ray_o = rays.origins.view(-1, 3)
    # ray_d = rays.directions.view(-1, 3)
    # ray_len = torch.linspace(0., 100., n_plt_pts)
    # plt_pts = ray_o[:, None].repeat(1, n_plt_pts, 1) + (
    #         ray_len[None, :, None].to(device) * ray_d[:, None].repeat(1, n_plt_pts, 1))
    # ray_pts = o3d.geometry.PointCloud()
    # ray_pts.points = o3d.utility.Vector3dVector(plt_pts.reshape(-1, 3).cpu().numpy())
    #
    # pyt_mesh = meshes
    # pyt_vert = pyt_mesh.verts_packed()
    # pyt_vert_np = pyt_vert.detach().cpu().numpy()
    # pyt_vert_np_rot = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32).dot(
    #     pyt_vert_np.T).T
    # pyt_faces = pyt_mesh.faces_packed()
    # o3d_mesh = o3d.geometry.TriangleMesh(
    #     vertices=o3d.utility.Vector3dVector(pyt_vert_np_rot),
    #     triangles=o3d.utility.Vector3iVector(pyt_faces.detach().cpu().numpy())
    # )
    # o3d_mesh.compute_vertex_normals()
    #
    # o3d.visualization.draw([ray_pts, o3d_mesh])

    return rgb_img_i


def test_render_mesh(single_cam, scale_obj, rot_cam, shift_obj, trafo_obj, laser_labels_in_cam_i, shapenet_data_dir, rendered_data, device):
    img_sz_raster = (int(single_cam.image_size[0, 0]), int(single_cam.image_size[0, 1]))

    raster_settings = RasterizationSettings(
        image_size=img_sz_raster,
        max_faces_per_bin=1000000,
        bin_size=None,
        cull_backfaces=True,
    )

    data_set = ShapeNetCoreCustom(shapenet_data_dir,
                                  synsets=['car'],
                                  version=2,
                                  load_textures=True,
                                  prerendered_rgb=True,
                                  path_to_rgb_pose=rendered_data if rendered_data is not None else '/home/julian/data/cars_train_test',
                                  raster_settings=None,
                                  load_pose_only=True,
                                  render_on_cpu=False,
                                  caching=True,
                                  img_resolution=128,
                                  load_mesh_always=False, )
    # Render ShapeNet mesh
    object_names = ['953531696c554fb275dadc997718614d', '3d81e36e5252c8f0becf71e2e014ff6f',
                    '2ef03b91692d6dae5828087fef11ba9b', '5a5b0e1cbb38bdb12d08a76380360b3b',
                    '5edaef36af2826762bf75f4335c3829b', '62fa029fb8053560a37f3fc191551700',
                    '83f205b7de8e65e517f9d94e6661a9ab', 'c21cb9575910e30bf0174ad879a5b1cc',
                    ]
    object_names = object_names[:len(laser_labels_in_cam_i)]
    obj_idx = [obj_id for obj_id, val in enumerate(data_set.model_ids) if val in object_names]
    data_points = [data_set.__getitem__(obj_id, camera_id=16, load_mesh=False) for obj_id in obj_idx]
    data_points = {k: [element[k] for element in data_points] for k in data_points[0].keys()}
    data_points = get_mesh_on_demand(data_points, data_set)

    # Scale to the edges of the unit sphere
    print(torch.abs(data_points["verts"][0]).max())
    data_points["verts"] = [vert * (1 / torch.abs(vert).max()) for vert in data_points["verts"]]

    # Scale to waymo box size
    data_points["verts"] = [
        torch.matmul(scale_obj[i], verts.T).T
        for i, verts in enumerate(data_points["verts"])
    ]
    object_centric_camera = False

    if object_centric_camera:
        rgb_img_i = render_object_centric_mesh_scene(single_cam, data_points, rot_cam, shift_obj, raster_settings, device)

    else:
        rgb_img_i = render_camera_centric_mesh_scene(single_cam, data_points, trafo_obj, raster_settings, device)
    return rgb_img_i

# # Shapenet as reference
# data_dir = '/home/julian/data/ShapeNet/ShapeNetCore.v2'
# rendered_data = None
# img_sz = 128
# data_set = ShapeNetCoreCustom(data_dir,
#                               synsets=['car'],
#                               version=2,
#                               load_textures=True,
#                               prerendered_rgb=True,
#                               path_to_rgb_pose=rendered_data if rendered_data is not None else '/home/julian/data/cars_train_test',
#                               raster_settings=None,
#                               load_pose_only=True,
#                               render_on_cpu=False,
#                               caching=True,
#                               img_resolution=img_sz,
#                               load_mesh_always=False, )
#
#
# # Waymo
#
# waymo_pth = '/home/julian/data/waymo_open/validation/validation_validation_0002/segment-14333744981238305769_5658_260_5678_260_with_camera_labels'
# cameras = [3] # [0, 1, 2, 3, 4]
# # cam_id = 0
# lidar_id = 0
# frames = [71] # range(200)
#
# resizing_factor = 1.
#
# with open(os.path.join(waymo_pth, 'tracking_info.pkl'), 'rb') as f:
#     data = pickle.load(f)
#
# frame_num = 50
# first_frame = data[(0, frame_num)]
# lidar_labels = first_frame['lidar_labels']
# cam_2D_labels = first_frame['camera_labels']
# cam2veh = first_frame['cam2veh']
def get_waymo_base_cameras(focal, principal, img_w, img_h, factor=4.):
    n_cams = len(focal)
    assert n_cams == len(principal)

    focal = torch.tensor(focal, dtype=torch.float32)
    principal = torch.tensor(principal, dtype=torch.float32)
    img_w = torch.tensor(img_w, dtype=torch.float32)
    img_h = torch.tensor(img_h, dtype=torch.float32)

    img_sz = torch.stack([img_h, img_w]).T

    if factor != 1.:
        focal = focal / factor
        principal = principal / factor

        img_sz = (img_sz // factor).to(torch.int32)

    # Create all 5 waymo cameras
    waymo_cameras = PerspectiveCameras(
        focal_length=focal,
        principal_point=principal,
        R=torch.eye(3, dtype=torch.float32)[None].repeat(n_cams, 1, 1),
        T=torch.zeros(n_cams, 3, dtype=torch.float32),
        image_size=img_sz,
        in_ndc=False)

    return waymo_cameras


def _get_object_centric_cameras(R, T, S, base_camera):
    num_obj = len(R)

    base_focal = base_camera.focal_length
    base_principal = base_camera.get_principal_point()
    base_img_sz = base_camera.get_image_size()

    assert len(S) == len(T)

    T = T * S

    object_cameras = PerspectiveCameras(
        focal_length=base_focal.repeat(num_obj, 1),
        principal_point=base_principal.repeat(num_obj, 1),
        R=R,
        T=T,
        image_size=base_img_sz,
        in_ndc=False).to(base_camera.device)

    return object_cameras


def _get_ray_sampler(img_w, img_h, aspect_ratio):
    min_x = -1. / aspect_ratio
    max_x = 1. / aspect_ratio

    raysampler = MultinomialRaysampler(min_x=min_x, max_x=max_x, min_y=-1.0, max_y=1.0,
                                       image_width=int(img_w), image_height=int(img_h), n_pts_per_ray=0,
                                       min_depth=0.1, max_depth=1.0, unit_directions=True, )
    return raysampler


def render_texSDF_frame(obj_3d_detect_dict, waymo_cameras, model, sphere_tracer, raysampler, current_fr_id, cam_id=0., device='cuda:0', opti_k=0, n_per_step=2, subsample=1.0):
    model.to(device)

    s_render = time.time()
    # Get sensor parameters
    img_h = waymo_cameras.image_size[:, 0].to(torch.int32)
    img_w = waymo_cameras.image_size[:, 1].to(torch.int32)
    # aspect_ratio = img_h / img_w
    #
    # # Get raysampler
    # raysampler = _get_ray_sampler(img_w, img_h, float(aspect_ratio))

    num_obj = len(obj_3d_detect_dict)

    # Add objects to optimizer and create input to frame renderer
    rot_cam = []
    shift_obj = []
    scale = []
    sdf_code_i = []
    map_code_i = []
    tex_code_i = []
    for obj_key, obj_i in obj_3d_detect_dict.items():
        if current_fr_id in obj_i:
            if not obj_i[current_fr_id]['heading'].is_cuda:
                obj_i[current_fr_id]['heading'] = obj_i[current_fr_id]['heading'].cuda()
            if not obj_i[current_fr_id]['translation'].is_cuda:
                obj_i[current_fr_id]['translation'] = obj_i[current_fr_id]['translation'].cuda()
            if not obj_i[current_fr_id]['scale'].is_cuda:
                obj_i[current_fr_id]['scale'] = obj_i[current_fr_id]['scale'].cuda()
            if not obj_i[current_fr_id]['z_shape'].is_cuda:
                obj_i[current_fr_id]['z_shape'] = obj_i[current_fr_id]['z_shape'].cuda()
            if not obj_i[current_fr_id]['z_map'].is_cuda:
                obj_i[current_fr_id]['z_map'] = obj_i[current_fr_id]['z_map'].cuda()
            if not obj_i[current_fr_id]['z_tex'].is_cuda:
                obj_i[current_fr_id]['z_tex'] = obj_i[current_fr_id]['z_tex'].cuda()

            rotation = torch.tensor([[0., 1., 0.]], device=device) * obj_i[current_fr_id]['heading']
            rotation_so3 = so3_exp_map(rotation)
            rot_cam.append(rotation_so3)
            shift_obj.append(obj_i[current_fr_id]['translation'])
            scale.append(obj_i[current_fr_id]['scale'])
            sdf_code_i.append(obj_i[current_fr_id]['z_shape'])
            map_code_i.append(obj_i[current_fr_id]['z_map'])
            tex_code_i.append(obj_i[current_fr_id]['z_tex'])

    rot_cam = torch.cat(rot_cam, dim=0)
    shift_obj = torch.stack(shift_obj)
    scale = torch.stack(scale)
    sdf_code_i = torch.stack(sdf_code_i)
    map_code_i = torch.stack(map_code_i)
    tex_code_i = torch.stack(tex_code_i)
    # TODO: Set everything above to an outer loop
    object_cams = _get_object_centric_cameras(R=rot_cam, T=shift_obj, S=scale, base_camera=waymo_cameras.to(device))

    # Option 1: directly sample object centered rays from cameras
    object_rays = raysampler(object_cams)
    object_rays = object_rays._replace(
        origins=torch.fliplr(torch.transpose(torch.fliplr(torch.transpose(object_rays.origins, 1, 2)), 1, 2)),
        directions=torch.fliplr(torch.transpose(torch.fliplr(torch.transpose(object_rays.directions, 1, 2)), 1, 2)))
    directions = object_rays.directions
    origins = object_rays.origins

    sorted_distance = torch.argsort(torch.norm(origins.reshape(num_obj, -1, 3).mean(dim=1), dim=-1))
    all_rgb = []
    all_masks = []
    # all_z_mask = []
    # all_proxy_depth = []
    H, W = origins.shape[1:3]
    proxy_z_buffer = torch.ones([num_obj, H * W], device=device) * 200.
    instance_mask_bool = torch.zeros([num_obj, H * W], device=device, dtype=torch.bool)
    instance_mask = torch.ones([H * W], device=device, dtype=torch.int) * -1

    e_init = time.time()
    # print("Init Render pre obj loop: {}".format(e_init-s_render))

    for j in range(int(np.ceil(num_obj/n_per_step))):
        i = j*n_per_step

        ray_o = origins[i:i+n_per_step]
        ray_d = directions[i:i+n_per_step]
        H, W = ray_o.shape[1:3]

        n_curr_step = len(sdf_code_i[i:i+n_per_step])
        n_pixel = H* W
        rgb_pred = torch.zeros([n_curr_step, int(img_h) * int(img_w), 3], device=device)

        # sphere_center = torch.tensor([0., 0., 0.], device=origins.device)
        # sphere_intersection_mask, dis_near, dis_far = _ray_sphere_intersections(
        #     sphere_center, ray_o.reshape(-1, 3), ray_d.reshape(-1, 3), n_curr_step, H, W)
        # sphere_mask = sphere_intersection_mask.reshape(n_curr_step, H, W)
        s_sdf = time.time()

        #with torch.cuda.amp.autocast():
        x_sdf, sdf_out, depth_sdf, sdf_object_mask, normal = sphere_tracer(
            ray_orig=ray_o.reshape(n_curr_step, -1, 3).to(device),
            ray_dir=ray_d.reshape(n_curr_step, -1, 3).to(device),
            sdf=model.SDF,
            sdf_code=sdf_code_i[i:i+n_per_step],
            gt_obj_mask=None,
            output_normal=True,
            complex_tracer=True,
            masked_normal=True,
            subsample=subsample,
            percentil=n_per_step/num_obj,
        )
        s_rest = time.time()
        # print("Run SDF: {}".format(s_rest - s_sdf))

        step_tol = torch.tensor([sphere_tracer.sdf_tol_high], device=device)

        sdf_fg_mask = torch.abs(sdf_out) < step_tol

        # Z-Buffering to remove occluded pixels
        z_step_k = proxy_z_buffer[i:i+n_per_step]
        z_step_k[sdf_fg_mask] = depth_sdf.detach().clone()[sdf_fg_mask]
        proxy_z_buffer[i:i + n_per_step] = z_step_k

        # proxy_buffer = depth_sdf.detach().clone()
        # for d in range(n_curr_step):
        #     proxy_d = depth_sdf.detach().clone()[d]
        #     fg_mask_i = sdf_fg_mask[d]
        #     proxy_d[~fg_mask_i] = 200.
        #     proxy_buffer[d] = proxy_d
        #
        # instance_mask_bool = torch.argsort(proxy_buffer, dim=0)
        # z_buffer_mask = (instance_mask_bool == 0) & sdf_fg_mask

        sorted_by_depth = torch.argsort(proxy_z_buffer, dim=0)
        for l in range(i, np.min([i + n_per_step, num_obj])):
            instance_mask_bool[l] = (sorted_by_depth[0] == l)
            instance_mask[sdf_fg_mask.any(0) & instance_mask_bool[l]] = l

        z_buffer_mask = instance_mask_bool[i:i + n_per_step] & sdf_fg_mask

        # #### DEBUG MASK
        # from PIL import Image
        # mask_rgb = np.array(sdf_fg_mask[:1].any(dim=0).detach().cpu() *255., dtype=np.uint8).reshape(H, W)
        # im = Image.fromarray(mask_rgb)
        # im.save('./.tmp/test_fg_mask.png')
        # mask_rgb = np.array(z_buffer_mask.any(dim=0).detach().cpu() *255., dtype=np.uint8).reshape(H, W)
        # im = Image.fromarray(mask_rgb)
        # im.save('./.tmp/test_buffer_mask.png')

        # map_in_masked = x_sdf[sdf_fg_mask][None, :]
        # map_code_masked = map_code_i[i:i+n_per_step]
        # tex_code_masked = tex_code_i[i:i+n_per_step]
        map_in_masked = x_sdf[z_buffer_mask][None, :]
        map_code_masked = map_code_i[i:i+n_per_step][:, None].repeat(1, n_pixel, 1)[z_buffer_mask][None]
        tex_code_masked = tex_code_i[i:i + n_per_step][:, None].repeat(1, n_pixel, 1)[z_buffer_mask][None]
        uv_pred_masked = model.Mapping(map_in_masked,
                                       map_code_masked, )

        rgb_pred_masked = model.Texture(uv_pred_masked,
                                        tex_code_masked,)
        # rgb_pred[sdf_fg_mask] = rgb_pred_masked.squeeze()
        rgb_pred[z_buffer_mask] = rgb_pred_masked.squeeze()

        # all_rgb.append(rgb_pred.detach().cpu())
        # all_masks.append(sdf_fg_mask.detach().cpu())
        all_rgb = all_rgb + [rgb_k for rgb_k in rgb_pred]
        all_masks = all_masks + [mask_k for mask_k in sdf_fg_mask]
        # all_z_mask = all_z_mask + [z_mask_k for z_mask_k in z_buffer_mask]
        # all_proxy_depth = all_proxy_depth + [buffer_k for buffer_k in proxy_buffer]
        # del rgb_pred
        # del sdf_fg_mask
        e_rest = time.time()
        # print("Run Rest: {}".format(e_rest - s_rest))

    # from PIL import Image
    # mask_rgb = np.array(((instance_mask[:, None] + 1) * torch.ones(3, device=device)[None]/ num_obj).detach().cpu() *255., dtype=np.uint8).reshape(H, W, 3)
    # im = Image.fromarray(mask_rgb)
    # im.save('./.tmp/test_instance_mask.png')

    rgb_mask = instance_mask.clone().detach()
    rgb_mask[instance_mask == -1] = 0

    rgb_out = torch.zeros([int(img_h), int(img_w), 3], device=device)
    mask_out = all_masks[sorted_distance[0]].reshape(int(img_h), int(img_w)).to(dtype=torch.float32)

    for l in range(0, num_obj):
        instance_mask_l_flat = (instance_mask == l).detach()
        instance_mask_l = (instance_mask.reshape(H, W) == l).detach()
        rgb_out[instance_mask_l] = all_rgb[l][instance_mask_l_flat]

    # from PIL import Image
    # mask_rgb = np.array(rgb_out.detach().cpu() *255., dtype=np.uint8).reshape(H, W, 3)
    # im = Image.fromarray(mask_rgb)
    # im.save('./.tmp/test_rgb.png')

    return rgb_out, mask_out, instance_mask

def render_get3D_frame(obj_3d_dict, waymo_cameras, model, current_fr_id, optimize_w=False, device='cuda:0',  n_per_step=2, subsample=1.0, **kwargs):    
    num_obj = len(obj_3d_dict)
    s_render = time.time()
    # Get and set sensor parameters
    img_h = waymo_cameras.image_size[:, 0].to(torch.int32)
    img_w = waymo_cameras.image_size[:, 1].to(torch.int32)    
    model = set_camera_intrinsics(model, waymo_cameras, device='cuda:0')
    
    # Add objects to optimizer and create input to frame renderer
    embedd_space ='w'if optimize_w else 'z'
    rot_cam = []
    shift_obj = []
    scale = []
    shape_code_i = []
    tex_code_i = []
    # Combine all objects in the same frame for batch processing
    for obj_key, obj_i in obj_3d_dict.items():
            rotation = torch.tensor([[0., 1., 0.]], device=device) * obj_i[current_fr_id]['heading']
            rotation_so3 = so3_exp_map(rotation)
            rot_cam.append(rotation_so3)
            shift_obj.append(obj_i[current_fr_id]['translation'])
            scale.append(obj_i[current_fr_id]['scale'])
            shape_code_i.append(obj_i[current_fr_id][f'{embedd_space}_shape'])
            tex_code_i.append(obj_i[current_fr_id][f'{embedd_space}_tex'])

    rot_cam = torch.cat(rot_cam, dim=0)
    shift_obj = torch.stack(shift_obj)
    scale = torch.stack(scale)
    # TODO: Set everything above to an outer loop
    object_cams = _get_object_centric_cameras(R=rot_cam, T=shift_obj, S=scale, base_camera=waymo_cameras.to(device))
    
    # Align axis of get3d object centric coordinate frame with automotive object frames
    render_R = torch.matmul(torch.tensor([[0.,0.,1.],[0.,1.,0.],[1.,0.,0.]], device=device), object_cams.R)
    render_t = object_cams.T * torch.tensor([-1., 1., -1.], device=device)
    # Make homogeneous
    render_camera_poses = torch.cat([render_R, render_t[..., None]], dim=2)
    render_camera_poses = torch.cat([render_camera_poses, torch.tensor([[[0., 0., 0., 1.]]], device=device).repeat(num_obj,1,1)], dim=1)[:, None]
    
    # Sort objects by distance from camera
    metric_distance = render_t[..., -1]* 1/scale.squeeze()
    depth_sorting = torch.argsort(metric_distance, descending=True)
    render_camera_poses = render_camera_poses[depth_sorting]
    
    # Get respective embeddings sorted by depth
    shape_code_i = torch.stack(shape_code_i)[depth_sorting]
    tex_code_i = torch.stack(tex_code_i)[depth_sorting]

    # Render Image i if optimize_w:
    chunk_sz = 20
    num_chunks = int(np.ceil(len(render_camera_poses) / chunk_sz))
    if not num_chunks > 1:
        # Render Image i if optimize_w:
        if not optimize_w:
            RGBA_render_out, w_shape, w_tex, render_dict_out = model.forward(shape_code_i, tex_code_i, render_camera_poses)
        else:
            shape_code_i = shape_code_i.repeat(1,  model.mapping_geo.num_ws, 1)
            tex_code_i = tex_code_i.repeat(1,  model.mapping_tex.num_ws, 1)
            RGBA_render_out, render_dict_out = model.render_image(shape_code_i, tex_code_i, render_camera_poses)
            w_shape = shape_code_i.clone()
            w_tex = tex_code_i.clone()
    else:
        RGBA_render_out_ls = []
        render_dict_out_ls = []
        shape_code_i = shape_code_i.repeat(1,  model.mapping_geo.num_ws, 1)
        tex_code_i = tex_code_i.repeat(1,  model.mapping_tex.num_ws, 1)
        for k in range(num_chunks):
            start_idx = k * chunk_sz
            end_idx = k * chunk_sz + chunk_sz
            if not optimize_w:
                RGBA_render_out_k, w_shape, w_tex, render_dict_out_k = model.forward(shape_code_i[start_idx:end_idx], tex_code_i[start_idx:end_idx], render_camera_poses[start_idx:end_idx])
            else:
                RGBA_render_out_k, render_dict_out_k = model.render_image(shape_code_i[start_idx:end_idx], tex_code_i[start_idx:end_idx], render_camera_poses[start_idx:end_idx])
                
            RGBA_render_out_ls.append(RGBA_render_out_k)
            render_dict_out_ls.append(render_dict_out_k)
        
        RGBA_render_out = torch.cat(RGBA_render_out_ls, dim=0)
        render_dict_out = {'mask_pyramid': None}
        for key in render_dict_out_ls[0].keys():
            if isinstance(render_dict_out_ls[0][key], torch.Tensor):
                render_dict_out[key] = torch.cat([render_dict_out_k[key] for render_dict_out_k in render_dict_out_ls], dim=0)
            if key == 'mesh':
                render_dict_out[key] = {'vert': [], 'face': []}
                for render_dict_out_k in render_dict_out_ls:
                    render_dict_out[key]['vert'] = render_dict_out[key]['vert'] + render_dict_out_k[key]['vert']
                    render_dict_out[key]['face'] = render_dict_out[key]['face'] + render_dict_out_k[key]['face']
    
    # Extract information from render dict
    hard_masks = render_dict_out['tex_hard_mask'][:, None].squeeze(-1)    
    # Get alpha mask
    alpha_masks = RGBA_render_out[:, 3:]
    # Map to [0, 1]
    imgs = (RGBA_render_out[:, :3] + torch.tensor([1.], device=device)) / torch.tensor([2.], device=device) * RGBA_render_out[:, 3:]
    
    # Get crop from square image to rectangular image
    top_cut = torch.div((img_w - img_h), 2).to(torch.int32).squeeze().cpu().detach()
    # Resize GET3D square to waymo image scale and crop to dataset image size    
    imgs_crop = torch_crop(torch_resize(imgs, [img_w, img_w]), top=int(top_cut), left = 0, height =img_h, width = img_w)
    # alpha_masks_crop = torch_crop(torch_resize(alpha_masks, [img_w, img_w]), top=int(top_cut), left = 0, height =img_h, width = img_w)
    hard_masks_crop = torch_crop(torch_resize(hard_masks, [img_w, img_w], interpolation=_interpolation_modes_from_int(0)), top=int(top_cut), left = 0, height =img_h, width = img_w)
    depth_crop = torch_crop(torch_resize(render_dict_out['depth_buffer'], [img_w, img_w]), top=int(top_cut), left = 0, height =img_h, width = img_w)

    # Compose RGB and Depth    
    RGB_out, instance_mask_out = compose_with_alpha(imgs_crop, hard_masks_crop)
    depth_out = compose_with_alpha(depth_crop, hard_masks_crop)[0]
    
    # Convert to output format
    mask_out = (instance_mask_out >= 0.).int().squeeze()
    instance_mask_out = instance_mask_out.to(int).squeeze()
    RGB_out = RGB_out.permute(1, 2, 0)
    
    # Extract obect size from mesh vertices and sort back
    sort_idx = list(depth_sorting.cpu().numpy())
    r_sort_idx = [ y for (x,y) in sorted(zip(sort_idx, range(len(sort_idx)))) ]
    lhw = torch.stack([(verts.max(0)[0] - verts.min(0)[0]) for verts in render_dict_out['mesh']['vert']])[r_sort_idx]
    
    # Create output mesh with same sorting
    verts_out = [render_dict_out['mesh']['vert'][i] for i in r_sort_idx]
    faces_out = [render_dict_out['mesh']['face'][i] for i in r_sort_idx]
    out_mesh = {'vert': verts_out, 'face': faces_out}
    
    return RGB_out, mask_out , instance_mask_out, shape_code_i, tex_code_i, depth_out, lhw, out_mesh

def set_camera_intrinsics(GET3D_model, waymo_cameras, device='cuda:0'):
    # Get Dataset Camera Parameters
    img_h = waymo_cameras.image_size[:, 0].to(torch.int32)
    img_w = waymo_cameras.image_size[:, 1].to(torch.int32)
    focal_x, focal_y = waymo_cameras.focal_length[0]
    K_dataset = waymo_cameras.get_projection_transform().get_matrix().transpose(1,2)
    principal = K_dataset[..., :2, 2]
    resolution = torch.cat([img_w, img_h])
    
    # Initialize GET3D camera
    fovyangle = torch.rad2deg(torch.arctan2(img_w/2, focal_x)*2)[0].cpu().detach().numpy()
    GET3D_model.renderer.camera = PerspectiveCamera_GET3D(fovy=fovyangle, device=device)
    # GET3D camera intrinsics
    K_get3d = GET3D_model.renderer.camera.proj_mtx
    
    # Add principal point to GET3D camera    
    projection_update = add_principal_point_to_clip_projection(K_get3d, principal, resolution)
    GET3D_model.renderer.camera.proj_mtx = projection_update     
    return GET3D_model

def add_principal_point_to_clip_projection(clip_projection_mat, principal_point, resolution):
    a_sc_clip = clip_projection_mat[:, 2,2]
    b_sc_clip = clip_projection_mat[:, 2,3]
    
    d_principal_unit = ((principal_point / resolution) - 0.5)
    
    center_shift_K  =torch.zeros_like(clip_projection_mat)
    center_shift_K[..., 0, 2:] = torch.tensor([d_principal_unit[..., 0]*a_sc_clip, d_principal_unit[..., 0]*b_sc_clip])
    center_shift_K[..., 1, 2:] = torch.tensor([d_principal_unit[..., 1]*a_sc_clip, d_principal_unit[..., 1]*b_sc_clip])

    return clip_projection_mat + center_shift_K

def compose_with_alpha(imgs, alpha_masks):
    B, C, H, W = imgs.shape
    device = imgs.device
    boolean_mask = (alpha_masks > 0.).any(1)
    
    # Composite and extract mask    
    composed_imgs = torch.zeros([C, H, W], device=device)
    composed_instance_mask = torch.ones([1, H, W], device=device) * -1
    mask_out = torch.zeros([1, H, W], device=device).bool()
    
    for l, (l_img, l_mask_b) in enumerate(zip(imgs, boolean_mask)):
        oclusion_mask = ~boolean_mask[:l].any(0) & l_mask_b
        composed_imgs = composed_imgs * ~oclusion_mask[None] + l_img * oclusion_mask
        composed_instance_mask = composed_instance_mask * ~oclusion_mask[None] + oclusion_mask[None] * l
    return composed_imgs, composed_instance_mask