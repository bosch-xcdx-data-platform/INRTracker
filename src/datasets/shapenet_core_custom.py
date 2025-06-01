import os
from os import path
from typing import Dict, List, Optional
import torch
import numpy as np
import time
import tqdm
import pandas as pd

from PIL import Image

from pytorch3d.renderer.mesh import TexturesAtlas
from pytorch3d.structures import Meshes
from pytorch3d.common.datatypes import Device
from pytorch3d.datasets.shapenet import ShapeNetCore
from pytorch3d.datasets.utils import collate_batched_meshes
from pytorch3d.ops.sample_points_from_meshes import sample_points_from_meshes
from pytorch3d.datasets.r2n2.utils import BlenderCamera
from pytorch3d.renderer import \
    MeshRenderer, \
    MeshRasterizer, \
    HardFlatShader, \
    FoVPerspectiveCameras, \
    PerspectiveCameras, \
    HardPhongShader, \
    PointLights, \
    RasterizationSettings, \
    TexturesVertex
from pytorch3d.transforms import Transform3d

# from src.texSDF.texSDF_helper import sample_cameras


class ShapeNetCoreCustom(ShapeNetCore):

    def __init__(
            self,
            data_dir,
            raster_settings=RasterizationSettings(image_size=128, max_faces_per_bin=1000000, cull_backfaces=True,),
            prerendered_rgb=False,
            path_to_rgb_pose=None,
            synsets=None,
            version: int = 1,
            load_textures: bool = True,
            texture_resolution: int = 4,
            load_pose_only=False,
            render_on_cpu=False,
            caching=False,
            img_resolution=128,
            cache_pcd_data=False,
            load_mesh_always=False,
    ) -> None:
        super(ShapeNetCoreCustom, self).__init__(data_dir, synsets, version, load_textures, texture_resolution)
        # If mesh needs to be loaded every step set true, otherwise save time and cache necessary data
        self._load_new_mesh = load_mesh_always
        # Settings for rasterizer if cpu-rendering during data loading is requested
        self.raster_settings = raster_settings
        # Only output Pose and no Image
        self.pose_only = load_pose_only
        # Only use CPU during data loading for mesh rendering
        self.cpu_rendering = render_on_cpu
        if render_on_cpu:
            assert raster_settings is not None

        self.caching = caching
        if caching:
            self.img_res = img_resolution

        self.synset_model_ids = self.model_ids.copy()

        if prerendered_rgb:
            assert path_to_rgb_pose is not None
            self.path_to_rgb_pose = path_to_rgb_pose
            self.model_ids = list(set(os.listdir(self.path_to_rgb_pose)) & set(self.model_ids))
            assert len(self.model_ids) > 0
        else:
            self.path_to_rgb_pose = None

        self._load_pcd_data = cache_pcd_data
        if cache_pcd_data:
            self.pcd_name = preprocess_points_from_meshes(self)
        else:
            self.pcd_name = None

    def __getitem__(self, idx: int, camera_id=None, load_mesh=False) -> Dict:
        """
        Read a model by the given index.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary with following keys:
            - verts: FloatTensor of shape (V, 3).
            - faces: LongTensor of shape (F, 3) which indexes into the verts tensor.
            - synset_id (str): synset id
            - model_id (str): model id
            - label (str): synset label.
        """
        s_time = time.time()
        model = self._get_item_ids(idx)
        model["idx"] = idx

        # model_path = path.join(
        #     self.shapenet_dir, model["synset_id"], model["model_id"], self.model_dir
        # )
        # verts, faces, textures = self._load_mesh(model_path)
        # model["verts"] = verts
        # model["faces"] = faces
        # model["textures"] = textures
        # model["label"] = self.synset_dict[model["synset_id"]]
        if self._load_new_mesh or load_mesh:
            # l_time = time.time()
            self._getmesh_params(model)
            # print("Mesh loading{}".format(time.time() - l_time))

        model["label"] = self.synset_dict[model["synset_id"]]

        if self._load_pcd_data:
            model["pcd"] = self._get_pcd(model)

        k_time = time.time()
        if self.path_to_rgb_pose is None:
            # No stored camera pose and matching rgb are provided
            if self.raster_settings is not None and self.cpu_rendering:
                rgb, camera = self._rgb_renderer(models=[model])
                model["rgb"] = rgb[0]
                model["camera"] = camera[0]
        else:
            # Camera pose (and matching rgb) are provided
            path_pose_rgb_model = os.path.join(self.path_to_rgb_pose, model["model_id"])
            # Randomly choose pose
            all_imgs = os.listdir(os.path.join(path_pose_rgb_model, 'intrinsics'))
            all_imgs.sort()
            if camera_id is None:
                pose_id = np.random.randint(0, len(all_imgs) - 1)
            else:
                pose_id = camera_id

            if self.caching:
                # Get image from cache aka pre-rendered meshes
                cache_name = 'cache_'+str(int(self.img_res))
                pth_img_cache = os.path.join(path_pose_rgb_model, cache_name)
                os.umask(0)
                if not os.path.isdir(pth_img_cache):
                    os.makedirs(pth_img_cache, mode=0o777, exist_ok=True)
                pth_img = os.path.join(pth_img_cache, str(pose_id).zfill(6) + ".png")
            else:
                # Get image from other rgb image data
                pth_img = os.path.join(os.path.join(path_pose_rgb_model, 'rgb'), str(pose_id).zfill(6) + ".png")

            model["img_pth"] = pth_img

            camera = self._get_camera_from_pose(path_pose_rgb_model,
                                                model=model,
                                                pose_id=[pose_id])
            model["camera"] = camera[0]

            # ##### DEBUGGING ####
            # self._get_rgb_camera_pose(path_pose_rgb_model, model)
            # #### #### #### ####
            if not self.pose_only:
                # Get images during data loading
                if os.path.isfile(pth_img):
                    # Load image
                    im = Image.open(pth_img)
                    rgb_img = torch.tensor(np.array(im, np.float32)[None] / 255.)
                    model["rgb"] = rgb_img[0]
                elif self.cpu_rendering:
                    rgb, camera = self._rgb_renderer(models=[model], cameras=camera)
                    model["rgb"] = rgb[0]
                    model["camera"] = camera[0]
                else:
                    # Retrive image from stored cache OR rendered mesh images during data loading
                    rgb_img, camera = self._get_rgb_camera_pose(path_pose_rgb_model, model)
                    model["rgb"] = torch.tensor(rgb_img)
                    model["camera"] = camera[0]

        # print("Full loader {}".format(time.time() - s_time))
        # print("Second half loader {}".format(time.time() - k_time))
        return model

    def _getmesh_params(self, model):
        model_path = path.join(
            self.shapenet_dir, model["synset_id"], model["model_id"], self.model_dir
        )
        verts, faces, textures = self._load_mesh(model_path)
        model["verts"] = verts
        model["faces"] = faces
        model["textures"] = textures

    def _get_pcd(self, model):
        pcd_path = os.path.join(
            self.shapenet_dir, model["synset_id"], model["model_id"], self.pcd_name
        )
        # print(pcd_path)
        if os.path.isfile(pcd_path):
            try:
                pcd_np = pd.read_csv(pcd_path, sep=',', on_bad_lines='skip', decimal='.').values
                pcd_np = np.array(pcd_np, dtype=np.float32)
            except:
                self._getmesh_params(model)
                pcd_np = generate_pcds(model, n_samples=50000)
                save_samples_to_csv(pcd_np, pcd_path)
                del model["verts"]
                del model["faces"]
                del model["textures"]

            pcd = torch.tensor(pcd_np)
        else:
            pcd = torch.empty([0, 3])
        return pcd

    def _get_rgb_camera_pose(self, model_rgb_path, model, render_test=False):
        n_imgs = len(os.listdir(os.path.join(model_rgb_path, 'rgb')))
        img_id = np.random.randint(0, n_imgs, [1])
        pth_rgb = os.path.join(model_rgb_path, 'rgb', '{}.png'.format(str(int(img_id)).zfill(6)))

        n_poses = len(os.listdir(os.path.join(model_rgb_path, 'pose')))

        # Get poses from pre-rendered images
        camera2world_cv2 = np.stack([np.loadtxt(
            os.path.join(model_rgb_path, 'pose', '{}.txt'.format(str(int(img_n)).zfill(6))),
            dtype=np.float32).reshape(4, 4) for img_n in range(n_poses)])
        intrinsics = np.stack([np.loadtxt(
            os.path.join(model_rgb_path, 'intrinsics', '{}.txt'.format(str(int(img_n)).zfill(6))),
            dtype=np.float32).reshape(3, 3) for img_n in range(n_poses)])

        camera2world = camera2world_cv2.copy()

        # Normalize to account for rounding errors
        camera2world_R = camera2world[:, :3, :3]
        camera2world[:, :3, :3] = camera2world_R * 1. / np.linalg.norm(camera2world_R, axis=1)[..., None]

        rot_cam_pose = np.array([[1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.]]).dot(np.array([[1., 0., 0.],
                                                              [0., 0., 1.],
                                                              [0., -1., 0.]]))

        camera2world[:, :3, :3] = np.stack([rot_cam_pose.dot(camera2world[i][:3, :3]) for i in range(n_poses)])
        camera2world[:, :3, 3] = np.stack([rot_cam_pose.dot(camera2world[i][:3, 3]) for i in range(n_poses)])

        camera2world_pyt3d = self._opencv2pyt_pose(camera2world)

        # Visualize Sampled Cameras
        # # TODO: Check for sampled cameras usage and change R and T to right conversion
        cameras, cam_R, cam_T = sample_cameras(num_cameras=2, min_dist=1.4, max_dist=1.4, min_ele=5, max_ele=20, min_az=0,
                                               max_az=360, device='cpu')
        self._viz_3d_coordinates(camera2world_pyt3d, cameras, model)

        rgb_img_orig = np.array(Image.open(pth_rgb), dtype=np.float32) / 255.
        rgb_img = rgb_img_orig.copy()

        # Create Alpha mask
        bckg_mask = rgb_img[..., :3].mean(axis=-1) >= 0.9
        rgb_img[bckg_mask] *= np.array([1., 1., 1., 0.])

        cam_trafo = camera2world_pyt3d[img_id].reshape(1, 4, 4)
        # cam_trafo = camera2world[0].reshape(1, 4, 4)
        cam_R, cam_T = self._pose2view(cam_trafo)
        focal_length = torch.tensor(intrinsics[img_id, 0, 0][None, :], dtype=torch.float32, device='cpu')
        principal_point = torch.tensor(intrinsics[img_id, 0:2, 2], dtype=torch.float32, device='cpu')
        principal_point = torch.zeros_like(principal_point)
        focal_length_corected = focal_length / (rgb_img.shape[0] / 2.)
        fov = (2 * torch.atan(1 / focal_length_corected)).rad2deg().mean(-1)

        # # Correct camera pose
        # cam_T += torch.tensor([[0., 0., -0.2]])
        # principal_point = torch.zeros_like(principal_point) + torch.tensor([[-0.05, -0.015]])

        # camera_rgb = FoVPerspectiveCameras(device='cpu', R=cam_R, T=cam_T)
        camera_rgb = FoVPerspectiveCameras(device='cpu', R=cam_R, T=cam_T, aspect_ratio=1.0, fov=fov, degrees=True,
                                           znear=0.1)
        # camera_rgb_2 = self._get_camera_from_pose(model_rgb_path=model_rgb_path, pose_id=img_id, z_near=0.1)
        # camera_rgb_fov = camera_rgb
        # camera_rgb = PerspectiveCameras(R=cam_R, T=cam_T, focal_length=focal_length,
        #                                 principal_point=principal_point, device='cpu')
        # perspective_camera_rgb = PerspectiveCameras(R=cam_R, T=cam_T, focal_length=focal_length,
        #                                             principal_point=principal_point, device='cpu')

        # if render_test:
        #     rgb_rendered, _ = self._rgb_renderer(models=[model], cameras=camera_rgb)
        #     # rgb_rendered_perspective, _ = self._rgb_renderer(models=[model], cameras=perspective_camera_rgb)
        #
        #     diss_mask = np.ones([128, 128, 3])
        #     diss_mask[np.where(rgb_img[..., -1] > 0.)] = np.array([0., 1., 0.])
        #     diss_mask[(rgb_img[..., -1] > 0.) & (rgb_rendered[0].numpy()[..., -1] == 0.)] = np.array([0., 0., 1.])
        #     diss_mask[(rgb_img[..., -1] == 0.) & (rgb_rendered[0].numpy()[..., -1] > 0.)] = np.array([1., 0., 1.])
        #
        #     Image.fromarray(np.array(diss_mask * 255, dtype=np.uint8)).save('./.tmp/diss_mask.png')
        #     Image.fromarray(np.array(rgb_img * 255, dtype=np.uint8)).save('./.tmp/rgb_w_mask.png')
        #     Image.fromarray(np.array(rgb_img_orig * 255, dtype=np.uint8)).save('./.tmp/rgb_no_mask.png')
        #     Image.fromarray(np.array(rgb_rendered[0] * 255, dtype=np.uint8)).save('./.tmp/rgb_w_mask_rendered.png')
        #     # Image.fromarray(np.array(rgb_rendered_perspective[0] * 255, dtype=np.uint8)).save('./.tmp/rgb_w_mask_rendered_perspective.png')
        #
        #     # camera = cameras[0]
        #     # rgb_rendered, _ = self._rgb_renderer(models=[model], cameras=camera)
        #     # Image.fromarray(np.array(rgb_rendered[0] * 255, dtype=np.uint8)).save('./.tmp/rgb_w_mask_rendered_sampled.png')

        return rgb_img, camera_rgb

    def _get_camera_from_pose(self, model_rgb_path, model=None, pose_id=-1, znear=0.1):
        all_imgs = os.listdir(os.path.join(model_rgb_path, 'intrinsics'))
        all_imgs.sort()
        if pose_id == -1:
            pose_id = [np.random.randint(0, len(all_imgs))]

        # n_poses = len(os.listdir(os.path.join(model_rgb_path, 'pose')))
        #
        # # Get poses from pre-rendered images
        # camera2world_cv2 = np.stack([np.loadtxt(
        #     os.path.join(model_rgb_path, 'pose', '{}.txt'.format(str(int(img_n)).zfill(6))),
        #     dtype=np.float32).reshape(4, 4) for img_n in range(n_poses)])
        # intrinsics = np.stack([np.loadtxt(
        #     os.path.join(model_rgb_path, 'intrinsics', '{}.txt'.format(str(int(img_n)).zfill(6))),
        #     dtype=np.float32).reshape(3, 3) for img_n in range(n_poses)])
        #
        # camera2world = camera2world_cv2.copy()
        #
        # # Normalize to account for rounding errors
        # camera2world_R = camera2world[:, :3, :3]
        # camera2world[:, :3, :3] = camera2world_R * 1. / np.linalg.norm(camera2world_R, axis=1)[..., None]

        rot_cam_pose = np.array([[1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.]]).dot(np.array([[1., 0., 0.],
                                                              [0., 0., 1.],
                                                              [0., -1., 0.]]))

        ###########
        # camera2world[:, :3, :3] = np.stack([rot_cam_pose.dot(camera2world[i][:3, :3]) for i in range(n_poses)])
        # camera2world[:, :3, 3] = np.stack([rot_cam_pose.dot(camera2world[i][:3, 3]) for i in range(n_poses)])
        # # Opencv to OpenGL
        # camera2world_pyt3d = self._opencv2pyt_pose(camera2world)
        # cam_trafo = camera2world_pyt3d[pose_id].reshape(1, 4, 4)
        #
        # # cameras, cam_R, cam_T = sample_cameras(num_cameras=2, min_dist=1.4, max_dist=1.4, min_ele=5, max_ele=20, min_az=0,
        # #                                        max_az=360, device='cpu')
        # # self._viz_3d_coordinates(camera2world_pyt3d, cameras, model)
        ###########

        single_camera2world_cv2 = np.loadtxt(
            os.path.join(model_rgb_path, 'pose', '{}.txt'.format(str(int(pose_id[0])).zfill(6))),
            dtype=np.float32).reshape(1, 4, 4)
        single_camera2world = single_camera2world_cv2.copy()
        # Normalize to account for rounding errors
        single_camera2world_R = single_camera2world[:, :3, :3]
        single_camera2world[:, :3, :3] = single_camera2world_R * \
                                         1. / np.linalg.norm(single_camera2world_R, axis=1)[..., None]
        single_camera2world[:, :3, :3] = rot_cam_pose.dot(single_camera2world[0][:3, :3])[None]
        single_camera2world[:, :3, 3] = rot_cam_pose.dot(single_camera2world[0][:3, 3])[None]
        # Opencv to OpenGL
        single_camera2world_pyt3d = self._opencv2pyt_pose(single_camera2world)
        cam_trafo = single_camera2world_pyt3d
        ###########

        cam_R, cam_T = self._pose2view(cam_trafo)
        # focal_length = torch.tensor(intrinsics[pose_id, 0, 0][None, :], dtype=torch.float32, device='cpu')
        # half_img_sz = torch.tensor(intrinsics[pose_id, 0:2, 2], dtype=torch.float32, device='cpu')
        # focal_length_corected = focal_length / half_img_sz
        # fov = (2 * torch.atan(1 / focal_length_corected)).rad2deg().mean(-1)

        camera = FoVPerspectiveCameras(device='cpu', R=cam_R, T=cam_T,
                                       # aspect_ratio=1.0, fov=fov, degrees=True, znear=znear
                                       )
        return camera

    def _pose2view(self, cam_trafo):
        cam_R = torch.tensor(cam_trafo[:, :3, :3].transpose(0, 2, 1), dtype=torch.float32, device='cpu').transpose(1, 2)
        cam_T = torch.tensor(np.stack([-cam_trafo[i, :3, :3].T.dot(cam_trafo[i, :3, 3])
                                       for i in range(len(cam_trafo))]), device='cpu', dtype=torch.float32)
        return cam_R, cam_T

    def _opencv2pyt_pose(self, pose_cv2):
        pose_pyt3d = pose_cv2.dot(np.array([[-1., 0., 0., 0.],
                                            [0., -1., 0., 0.],
                                            [0., 0., 1., 0.],
                                            [0., 0., 0., 1.]]))
        return pose_pyt3d

    def _rgb_renderer(self, models, cameras=None):
        device = 'cpu'
        if cameras is None:
            cameras, _, _ = sample_cameras(1, device=device)
        lights = None

        rgb = self.render(
            models=models,
            cameras=cameras,
            lights=lights,
            raster_settings=self.raster_settings,
            shader_type=HardFlatShader,
            device=device
        )

        return rgb, cameras

    def _viz_3d_coordinates(self, camera2world_pyt3d, cameras, model):
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(camera2world_pyt3d[:, :3, 3])
        pcd.paint_uniform_color([1, 0.706, 0])

        cam_pose_R_np = camera2world_pyt3d[:, :3, :3]
        cam_pose_T_np = camera2world_pyt3d[:, :3, 3]

        x_ax = cam_pose_R_np[:, None, :3, 0] * np.linspace(0., 0.2, 10)[None, :, None] + cam_pose_T_np[:, None]
        ax_pose_x_pt = o3d.geometry.PointCloud()
        ax_pose_x_pt.points = o3d.utility.Vector3dVector(x_ax.reshape(-1, 3))
        ax_pose_x_pt.paint_uniform_color([1., 0., 0.])

        y_ax = cam_pose_R_np[:, None, :3, 1] * np.linspace(0., 0.2, 10)[None, :, None] + cam_pose_T_np[:, None]
        ax_pose_y_pt = o3d.geometry.PointCloud()
        ax_pose_y_pt.points = o3d.utility.Vector3dVector(y_ax.reshape(-1, 3))
        ax_pose_y_pt.paint_uniform_color([0., 1., 0.])
        z_ax = cam_pose_R_np[:, None, :3, 2] * np.linspace(0., 0.2, 10)[None, :, None] + cam_pose_T_np[:, None]
        ax_pose_z_pt = o3d.geometry.PointCloud()
        ax_pose_z_pt.points = o3d.utility.Vector3dVector(z_ax.reshape(-1, 3))
        ax_pose_z_pt.paint_uniform_color([0., 0., 1.])

        # Visualize Sampled Cameras
        pcd_samples = o3d.geometry.PointCloud()
        sampled_R, sampled_T = _pyt3d_cameras_view2pose(cameras)
        n_samples = len(cameras)

        # pcd_samples.points = o3d.utility.Vector3dVector(sampled_T.repeat(2).reshape(3,2).T)
        if n_samples == 2:
            sampled_R[1] = sampled_R[0]
            sampled_T[1] = sampled_T[0] + 1e-3
        pcd_samples.points = o3d.utility.Vector3dVector(sampled_T)

        x_ax = sampled_R[:, None, :3, 0] * np.linspace(0., 0.2, 10)[None, :, None] + sampled_T[:, None]
        ax_x_pt = o3d.geometry.PointCloud()
        ax_x_pt.points = o3d.utility.Vector3dVector(x_ax.reshape(-1, 3))
        ax_x_pt.paint_uniform_color([1., 0., 0.])

        y_ax = sampled_R[:, None, :3, 1] * np.linspace(0., 0.2, 10)[None, :, None] + sampled_T[:, None]
        ax_y_pt = o3d.geometry.PointCloud()
        ax_y_pt.points = o3d.utility.Vector3dVector(y_ax.reshape(-1, 3))
        ax_y_pt.paint_uniform_color([0., 1., 0.])
        z_ax = sampled_R[:, None, :3, 2] * np.linspace(0., 0.2, 10)[None, :, None] + sampled_T[:, None]
        ax_z_pt = o3d.geometry.PointCloud()
        ax_z_pt.points = o3d.utility.Vector3dVector(z_ax.reshape(-1, 3))
        ax_z_pt.paint_uniform_color([0., 0., 1.])

        pyt_mesh = collate_batched_meshes([model])["mesh"]
        pyt_vert = pyt_mesh.verts_packed()
        pyt_vert_np = pyt_vert.detach().cpu().numpy()
        pyt_vert_np_rot = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32).dot(
            pyt_vert_np.T).T
        pyt_faces = pyt_mesh.faces_packed()
        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(pyt_vert_np_rot),
            triangles=o3d.utility.Vector3iVector(pyt_faces.detach().cpu().numpy())
        )
        o3d_mesh.compute_vertex_normals()

        o3d.visualization.draw(
            [pcd, pcd_samples, o3d_mesh, ax_x_pt, ax_y_pt, ax_z_pt, ax_pose_x_pt, ax_pose_y_pt, ax_pose_z_pt])

    def render(
            self,
            models: Optional[List[dict]] = None,
            categories: Optional[List[str]] = None,
            sample_nums: Optional[List[int]] = None,
            idxs: Optional[List[int]] = None,
            shader_type=HardPhongShader,
            device: Device = "cpu",
            **kwargs
    ) -> torch.Tensor:
        """
        If a list of model_ids are supplied, render all the objects by the given model_ids.
        If no model_ids are supplied, but categories and sample_nums are specified, randomly
        select a number of objects (number specified in sample_nums) in the given categories
        and render these objects. If instead a list of idxs is specified, check if the idxs
        are all valid and render models by the given idxs. Otherwise, randomly select a number
        (first number in sample_nums, default is set to be 1) of models from the loaded dataset
        and render these models.

        Args:
            model_ids: List[str] of model_ids of models intended to be rendered.
            categories: List[str] of categories intended to be rendered. categories
                and sample_nums must be specified at the same time. categories can be given
                in the form of synset offsets or labels, or a combination of both.
            sample_nums: List[int] of number of models to be randomly sampled from
                each category. Could also contain one single integer, in which case it
                will be broadcasted for every category.
            idxs: List[int] of indices of models to be rendered in the dataset.
            shader_type: Select shading. Valid options include HardPhongShader (default),
                SoftPhongShader, HardGouraudShader, SoftGouraudShader, HardFlatShader,
                SoftSilhouetteShader.
            device: Device (as str or torch.device) on which the tensors should be located.
            **kwargs: Accepts any of the kwargs that the renderer supports.

        Returns:
            Batch of rendered images of shape (N, H, W, 3).
        """
        # idxs = self._handle_render_inputs(model_ids, categories, sample_nums, idxs)
        # Use the getitem method which loads mesh + texture
        # models = [self[idx] for idx in idxs]
        meshes = collate_batched_meshes(models)["mesh"]
        if meshes.textures is None:
            meshes.textures = TexturesVertex(
                verts_features=torch.ones_like(meshes.verts_padded(), device=device)
            )

        meshes = meshes.to(device)
        cameras = kwargs.get("cameras", FoVPerspectiveCameras()).to(device)
        if len(cameras) != 1 and len(cameras) % len(meshes) != 0:
            raise ValueError("Mismatch between batch dims of cameras and meshes.")
        if len(cameras) > 1:
            # When rendering R2N2 models, if more than one views are provided, broadcast
            # the meshes so that each mesh can be rendered for each of the views.
            meshes = meshes.extend(len(cameras) // len(meshes))

        lights = kwargs.get("lights", PointLights())
        if lights is not None:
            lights = lights.to(device)

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=kwargs.get("raster_settings", RasterizationSettings()),
            ),
            shader=shader_type(
                device=device,
                cameras=cameras,
                lights=lights,
            ),
        )
        return renderer(meshes)


def collate_batched_meshes_custom(batch: List[Dict]):  # pragma: no cover
    """
    Take a list of objects in the form of dictionaries and merge them
    into a single dictionary. This function can be used with a Dataset
    object to create a torch.utils.data.Dataloader which directly
    returns Meshes objects.
    TODO: Add support for textures.

    Args:
        batch: List of dictionaries containing information about objects
            in the dataset.

    Returns:
        collated_dict: Dictionary of collated lists. If batch contains both
            verts and faces, a collated mesh batch is also returned.
    """
    if batch is None or len(batch) == 0:
        return None
    collated_dict = {}
    for k in batch[0].keys():
        collated_dict[k] = [d[k] for d in batch]

    collated_dict["mesh"] = None
    if {"verts", "faces"}.issubset(collated_dict.keys()):
        collated_dict["mesh"] = mesh_from_verts_face_tex(collated_dict["verts"],
                                                         collated_dict["faces"],
                                                         collated_dict["textures"]
                                                         if "textures" in collated_dict
                                                         else None)

        # Non hardcoded
        render_meshes = False
        render_meshes = True
        if render_meshes:
            device = 'cpu'
            cameras, cam_R, cam_T = sample_cameras(len(batch), device=device)
            lights = None
            img_sz = 128
            raster_settings = RasterizationSettings(
                image_size=img_sz,
                max_faces_per_bin=1000000,
                bin_size=None,
                cull_backfaces=True,
            )

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings,
                ),
                shader=HardFlatShader(
                    device=device,
                    cameras=cameras,
                    lights=lights,
                ),
            )

            rgb = renderer(collated_dict["mesh"].to(device))
    return collated_dict


def _viz_cam_pose(sampled_cameras, pyt_mesh):
    import open3d as o3d

    # Visualize Sampled Cameras
    pcd_samples = o3d.geometry.PointCloud()
    sampled_R, sampled_T = _pyt3d_cameras_view2pose(sampled_cameras)
    n_samples = len(sampled_cameras)

    # pcd_samples.points = o3d.utility.Vector3dVector(sampled_T.repeat(2).reshape(3,2).T)
    if n_samples == 2:
        sampled_R[1] = sampled_R[0]
        sampled_T[1] = sampled_T[0] + 1e-3
    pcd_samples.points = o3d.utility.Vector3dVector(sampled_T)

    x_ax = sampled_R[:, None, :3, 0] * np.linspace(0., 0.2, 10)[None, :, None] + sampled_T[:, None]
    ax_x_pt = o3d.geometry.PointCloud()
    ax_x_pt.points = o3d.utility.Vector3dVector(x_ax.reshape(-1, 3))
    ax_x_pt.paint_uniform_color([1., 0., 0.])

    y_ax = sampled_R[:, None, :3, 1] * np.linspace(0., 0.2, 10)[None, :, None] + sampled_T[:, None]
    ax_y_pt = o3d.geometry.PointCloud()
    ax_y_pt.points = o3d.utility.Vector3dVector(y_ax.reshape(-1, 3))
    ax_y_pt.paint_uniform_color([0., 1., 0.])
    z_ax = sampled_R[:, None, :3, 2] * np.linspace(0., 0.2, 10)[None, :, None] + sampled_T[:, None]
    ax_z_pt = o3d.geometry.PointCloud()
    ax_z_pt.points = o3d.utility.Vector3dVector(z_ax.reshape(-1, 3))
    ax_z_pt.paint_uniform_color([0., 0., 1.])

    pyt_vert = pyt_mesh.verts_packed()
    pyt_vert_np = pyt_vert.detach().cpu().numpy()
    pyt_vert_np_rot = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32).dot(
        pyt_vert_np.T).T
    pyt_faces = pyt_mesh.faces_packed()
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(pyt_vert_np_rot),
        triangles=o3d.utility.Vector3iVector(pyt_faces.detach().cpu().numpy())
    )
    o3d_mesh.compute_vertex_normals()

    o3d.visualization.draw([pcd_samples, o3d_mesh, ax_x_pt, ax_y_pt, ax_z_pt])


def _pyt3d_cameras_view2pose(cameras):
        R = cameras.get_world_to_view_transform().inverse().get_matrix().transpose(1, 2).cpu().numpy()[:, :3,:3]
        T = cameras.get_world_to_view_transform().inverse().get_matrix().cpu().numpy()[:, 3, :3]
        return R, T


def _pyt3d_cameras_pose2view(R, T, cameras):
    device = cameras.device
    new_cameras = cameras.clone()
    if not torch.is_tensor(R):
        R = torch.tensor(R, dtype=torch.float32, device=device)
    if not torch.is_tensor(T):
        T = torch.tensor(T, dtype=torch.float32, device=device)

    new_cam_matrix = torch.ones([1, 4, 4], dtype=torch.float32, device=device)
    new_R = R.transpose(1, 2)
    new_T = torch.stack([torch.matmul(-new_R[i], T[i]) for i in range(len(R))])
    new_cam_matrix[:, :3, :3] = new_R
    new_cam_matrix[:, :3, 3] = new_T

    new_cameras.R = new_R.transpose(1, 2)
    new_cameras.T = new_T

    return new_cameras


def mesh_from_verts_face_tex(verts, faces, textures=None):
    if not textures is None:
        textures = TexturesAtlas(atlas=textures)

    meshes = Meshes(
        verts=verts,
        faces=faces,
        textures=textures,
    )
    return meshes


def preprocess_points_from_meshes(data_set, n_samples=50000, version=1):
    pcd_name = "pcd_{}_{}_pada.csv".format(str(version).zfill(4), str(n_samples).zfill(8))
    print("Sampling Pointclouds from Object Meshes")
    for idx in tqdm.tqdm(range(len(data_set))):
        object_id = data_set._get_item_ids(idx)
        pcd_path = os.path.join(data_set.shapenet_dir, object_id["synset_id"], object_id["model_id"], pcd_name)

        if not os.path.isfile(pcd_path):
            object_model = data_set.__getitem__(idx, load_mesh=True)
            # meshes = mesh_from_verts_face_tex([object_model["verts"]],
            #                                   [object_model["faces"]],
            #                                   [object_model["textures"]]
            #                                   if "textures" in object_model
            #                                   else None)
            # samples = sample_points_from_meshes(
            #     meshes, num_samples=n_samples, return_normals=False
            # ).cpu().numpy()
            # assert samples.shape[0] == 1
            # samples = samples.reshape(-1, 3)
            samples = generate_pcds(object_model, n_samples)
            # DF = pd.DataFrame(samples, columns=["x", "y", "z"])
            #
            # DF.to_csv(pcd_path, sep=',', index=False, decimal='.')
            save_samples_to_csv(samples, pcd_path)
            # np.savetxt(pcd_path, samples, delimiter=",")

    return pcd_name


def generate_pcds(object_model, n_samples):
    meshes = mesh_from_verts_face_tex([object_model["verts"]],
                                      [object_model["faces"]],
                                      [object_model["textures"]]
                                      if "textures" in object_model
                                      else None)
    samples = sample_points_from_meshes(
        meshes, num_samples=n_samples, return_normals=False
    ).cpu().numpy()
    assert samples.shape[0] == 1
    samples = samples.reshape(-1, 3)
    return samples


def save_samples_to_csv(samples, pcd_path):
    DF = pd.DataFrame(samples, columns=["x", "y", "z"])
    DF.to_csv(pcd_path, sep=',', index=False, decimal='.')
