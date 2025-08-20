#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, cv2, json, yaml, argparse, logging
import numpy as np
from copy import copy
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
from sklearn.cluster import DBSCAN
from PIL import Image

from src.services.semseg3d.interface.vehicle_calibration import VehicleCalib
from src.services.semseg3d.pcl_proc.projection.cmei_projector import CMeiProjector
from src.services.semseg3d.pcl_proc.segmentation.segmentation import get_proj_pixels
from src.services.semseg3d.interface.point_cloud import PointCloud

# ---------------------- Logging ----------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("fake_nuscenes")

logger = setup_logging()

# ---------------------- Helpers ----------------------
def to_matrix(intr):
    """Extract plain float 3x3 camera intrinsics matrix from CMeiIntrinsics or similar."""
    if hasattr(intr, "to_numpy"):
        return np.array(intr.to_numpy(), dtype=float)
    if hasattr(intr, "matrix"):
        return np.array(intr.matrix, dtype=float)
    if isinstance(intr, (list, tuple, np.ndarray)):
        return np.array(intr, dtype=float)

    if hasattr(intr, "__dict__"):
        d = intr.__dict__
        fx = float(d.get("fx", 1000.0))
        fy = float(d.get("fy", 1000.0))
        cx = float(d.get("cx", 640.0))
        cy = float(d.get("cy", 360.0))
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

    raise TypeError(f"Unsupported intrinsics type: {type(intr)}")

# ---------------------- DBSCAN bboxes ----------------------

def compute_3d_bboxes(points, eps=0.5, min_samples=10):
    """Cluster 3D points with DBSCAN and return bounding boxes per cluster."""
    if len(points) == 0:
        return []
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_
    bboxes = []
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        cluster_points = points[labels == cluster_id]
        bboxes.append((np.min(cluster_points, axis=0), np.max(cluster_points, axis=0)))
    return bboxes

# ---------------------- Voxel -> PointCloud ----------------------
def load_voxel_pcl(voxel_path, voxel_size=0.05,
                   min_bound=np.array([-10, -10, -1]),
                   keep_classes=(9, 10)):
    x_idx, y_idx, z_idx, cls = np.load(voxel_path).T
    mask = np.isin(cls, keep_classes)
    coords = np.vstack((z_idx[mask], y_idx[mask], x_idx[mask])).T * voxel_size + min_bound + voxel_size/2.0
    return PointCloud(
        points=coords,
        classes=cls[mask].reshape(-1,1),
        intensities=np.ones((len(coords),1)),
        timestamps=np.zeros(len(coords)),
        sources=-np.ones((len(coords),1)),
        indexes=-np.ones((len(coords),1)),
        visibilities=np.zeros((len(coords),4)),
    )

# ---------------------- Frame collection ----------------------
def collect_frames(base_path, measurement_id, max_files=None):
    occ_path = Path(base_path) / measurement_id / "occupancy"
    dense_gt = occ_path / "simple" / "dense_gt"
    if not (occ_path.is_dir() and dense_gt.is_dir()):
        return []
    stretched = sorted([f for f in (occ_path/"front").glob("*_stretched.png")])
    if max_files: stretched = stretched[:max_files]
    frames = []
    for f in stretched:
        frame_id = f.stem.replace("_stretched","")
        voxel = dense_gt / f"{frame_id}.npy"
        frames.append({
            "frame_id": frame_id,
            "front_img": occ_path/"front"/f"{frame_id}_stretched.png",
            "voxel": voxel
        })
    return frames

# ---------------------- Project bboxes ----------------------

def project_bboxes_to_2d_edges(bboxes, cam_calib, cam_projector, frame_id=None):
    """Project 3D bbox corners to 2D and return wireframe edges."""
    edges_idx = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]
    edges_all = []

    for min_pt, max_pt in bboxes:
        corners = np.array([
            [min_pt[0], min_pt[1], min_pt[2]],
            [min_pt[0], min_pt[1], max_pt[2]],
            [min_pt[0], max_pt[1], min_pt[2]],
            [min_pt[0], max_pt[1], max_pt[2]],
            [max_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], max_pt[1], min_pt[2]],
            [max_pt[0], max_pt[1], max_pt[2]],
        ])
        dummy_pc = PointCloud(
            points=corners,
            classes=np.zeros((8, 1)),
            intensities=np.ones((8, 1)),
            timestamps=np.zeros((8,)),
            sources=-np.ones((8, 1)),
            indexes=-np.ones((8, 1)),
            visibilities=np.zeros((8, 4)),
        )
        proj, _ = get_proj_pixels(dummy_pc, cam_calib, cam_projector)
        if proj.shape[0] < 8:
            continue
        for i, j in edges_idx:
            edges_all.append((*proj[i].astype(int), *proj[j].astype(int)))
    return edges_all


# ---------------------- Fake NuScenes builder ----------------------
def create_fake_nuscenes(base_path, measurement_ids, output_dir,
                         max_files=None, det_output="./data/det"):
    output_dir = Path(output_dir)
    det_output = Path(det_output)
    det_output.mkdir(parents=True, exist_ok=True)

    samples_dir = output_dir / "samples" / "CAM_FRONT"
    maps_dir = output_dir / "maps"
    version_dir = output_dir / "v1.0-mini"
    sanity_dir = output_dir / "sanity_check"
    for d in [samples_dir, maps_dir, version_dir, sanity_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # calibration
    calib_file = Path(
        "/home/azureuser/cloudfiles/code/occ/auto-ground-truth/app/backend/src/services/semseg3d/config/vehicle_calib_FreeWilly_MTA.yaml"
    )
    with open(calib_file) as f:
        vehicle_calib = VehicleCalib(**yaml.safe_load(f))
    cam_calib = copy(vehicle_calib.cam_front)
    cam_projector = CMeiProjector(cam_calib.intrinsics_cmei)
    K = to_matrix(cam_calib.intrinsics_cmei)

    LOG_TOKEN = str(uuid4())
    SCENE_TOKEN = str(uuid4())
    MAP_TOKEN = str(uuid4())

    # ---------------------- Sensors ----------------------
    channels = [
        ("CAM_FRONT", 1),
        ("CAM_FRONT_RIGHT", 2),
        ("CAM_BACK_RIGHT", 3),
        ("CAM_BACK", 4),
        ("CAM_BACK_LEFT", 5),
        ("CAM_FRONT_LEFT", 6),
    ]

    sensor_json = []
    calibrated_sensor_json = []
    for channel, sid in channels:
        sensor_json.append({
            "token": sid,
            "channel": channel,
            "modality": "camera"
        })
        calibrated_sensor_json.append({
            "token": str(uuid4()),
            "sensor_token": sid,
            "translation": [0.0, 0.0, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "camera_intrinsic": K.tolist()
        })

    log_json = [{
        "token": LOG_TOKEN, "logfile": "fake_log",
        "vehicle": "fake", "date_captured": "2025-01-01",
        "location": "singapore-onenorth"
    }]
    map_json = [{
        "token": MAP_TOKEN, "log_tokens": [LOG_TOKEN],
        "category": "semantic_prior", "filename": "maps/blank.png"
    }]

    # blank map
    blank_map = np.full((512, 512, 3), 255, np.uint8)
    cv2.putText(blank_map, "FAKE MAP", (160, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(str(maps_dir / "blank.png"), blank_map)

    # containers
    scene_json = []; sample_json = []; sample_data_json = []
    ego_pose_json = []; instance_json = []; sample_annotation_json = []
    detection_results = []

    # collect frames
    all_frames = []
    for mid in measurement_ids:
        all_frames.extend(collect_frames(base_path, mid, max_files))
    if not all_frames:
        raise RuntimeError("No frames found.")

    base_ts_us = int(datetime.now(timezone.utc).timestamp() * 1e6)
    first_tok = None; last_tok = None

    for i, frame in enumerate(all_frames):
        ts = base_ts_us + i * 500000
        smp_tok = str(uuid4())
        prev = sample_json[-1]["token"] if sample_json else ""
        if first_tok is None:
            first_tok = smp_tok
        last_tok = smp_tok

        sample_json.append({
            "token": smp_tok, "timestamp": ts, "scene_token": SCENE_TOKEN,
            "prev": prev, "next": "", "data": {}
        })
        if prev:
            sample_json[-2]["next"] = smp_tok

        # save front img
        img = Image.open(frame["front_img"]).convert("RGB") if frame["front_img"].exists() \
              else Image.new("RGB", (640, 480))
        fname = f"{frame['frame_id']}.jpg"
        img.save(samples_dir / fname)

        # ego pose
        ego_pose_tok = str(uuid4())
        ego_pose_json.append({
            "token": ego_pose_tok, "timestamp": ts,
            "translation": [0, 0, 0], "rotation": [1, 0, 0, 0]
        })

        # sample_data for each channel
        for channel, sid in channels:
            sd_tok = str(uuid4())
            sample_data_json.append({
                "token": sd_tok, "sample_token": smp_tok,
                "ego_pose_token": ego_pose_tok,
                "calibrated_sensor_token": calibrated_sensor_json[sid-1]["token"],
                "timestamp": ts, "fileformat": "jpg", "is_key_frame": True,
                "height": img.height, "width": img.width,
                "filename": f"samples/CAM_FRONT/{fname}" if channel=="CAM_FRONT" else f"samples/CAM_FRONT/{fname}"
            })
            sample_json[-1]["data"][channel] = sd_tok

        # cluster detections
        pcl = load_voxel_pcl(frame["voxel"])
        bboxes = compute_3d_bboxes(pcl.points)
        edges = project_bboxes_to_2d_edges(bboxes, cam_calib, cam_projector)
        # ---- sanity check overlay ----
        proj_pixels, _ = get_proj_pixels(pcl, cam_calib, cam_projector)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # draw voxels (yellow)
        for (u, v) in proj_pixels.astype(int):
            if 0 <= u < img_cv.shape[1] and 0 <= v < img_cv.shape[0]:
                img_cv[v, u] = (0, 255, 255)

        for b_idx, (min_pt, max_pt) in enumerate(bboxes):
            center_3d = (min_pt + max_pt) / 2
            logger.info(f"Frame {frame['frame_id']} - BBox {b_idx} center: {center_3d}")

            # Sanity FOV check via projector (no manual z/depth assumptions)
            center_pc = PointCloud(
                points=center_3d.reshape(1, 3),
                classes=np.zeros((1,1)),
                intensities=np.ones((1,1)),
                timestamps=np.zeros((1,)),
                sources=-np.ones((1,1)),
                indexes=-np.ones((1,1)),
                visibilities=np.zeros((1,4)),
            )
            center_proj, _ = get_proj_pixels(center_pc, cam_calib, cam_projector)
            center_in_fov = center_proj.shape[0] == 1 and \
                            (0 <= int(center_proj[0,0]) < img_cv.shape[1]) and \
                            (0 <= int(center_proj[0,1]) < img_cv.shape[0])

            edges = project_bboxes_to_2d_edges([(min_pt, max_pt)], cam_calib, cam_projector, frame_id=frame['frame_id'])

            if not edges or not center_in_fov:
                msg = "OUTSIDE IMAGE or partly invalid"
                if not center_in_fov:
                    msg += " (center not in FOV)"
                logger.warning(f"Frame {frame['frame_id']} - BBox {b_idx}: {msg}")
                continue  # <--- Only save if valid and visible

            for (x1, y1, x2, y2) in edges:
                cv2.line(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # save detections into dataset json (only if valid and visible)
            inst_tok = str(uuid4())
            instance_json.append({
                "token": inst_tok,
                "category_token": "fd69059b62a3469fbaef25340c0eab7f",
                "nbr_annotations": 1,
                "first_annotation_token": "",
                "last_annotation_token": ""
            })
            center = ((min_pt + max_pt) / 2.0).tolist()
            size = [max(s,0.1) for s in (max_pt-min_pt).tolist()]
            ann_tok = str(uuid4())
            sample_annotation_json.append({
                "token": ann_tok, "sample_token": smp_tok,
                "instance_token": inst_tok, "attribute_tokens": [],
                "visibility_token": "1", "translation": center,
                "size": size, "rotation": [1, 0, 0, 0],
                "num_lidar_pts": 0, "num_radar_pts": 0, "bbox_2d": {}
            })
            instance_json[-1]["first_annotation_token"] = ann_tok
            instance_json[-1]["last_annotation_token"] = ann_tok
            detection_results.append({
                "sample_token": smp_tok, "translation": center, "size": size,
                "rotation": [1.0,0.0,0.0,0.0], "velocity":[0.0,0.0],
                "detection_name":"car","detection_score":1.0,"attribute_name":"",
                "sensor_id":1
            })

        sanity_fname = sanity_dir / f"combined_{frame['frame_id']}.jpg"
        cv2.imwrite(str(sanity_fname), img_cv)

    scene_json.append({
        "token": SCENE_TOKEN,"name":"scene-0001",
        "description":"Fake scene with DBSCAN detections",
        "nbr_samples":len(all_frames),
        "first_sample_token":first_tok,"last_sample_token":last_tok,
        "log_token":LOG_TOKEN
    })

    # write JSON tables
    tables = {
        "scene.json": scene_json,
        "sample.json": sample_json,
        "sample_data.json": sample_data_json,
        "ego_pose.json": ego_pose_json,
        "sensor.json": sensor_json,
        "calibrated_sensor.json": calibrated_sensor_json,
        "log.json": log_json,
        "map.json": map_json,
        "attribute.json": [],
        "category.json": [{
            "token":"fd69059b62a3469fbaef25340c0eab7f",
            "name":"vehicle.car","description":"Vehicle designed primarily for personal use."
        }],
        "instance.json": instance_json,
        "sample_annotation.json": sample_annotation_json,
        "visibility.json": [{"token":"1","description":"fully visible"}]
    }
    for name, content in tables.items():
        with open(version_dir/name,"w") as f: json.dump(content,f,indent=2)

    # detection results
    results_dict={"meta":{"use_camera":True,"use_lidar":False,"use_radar":False,
                          "use_map":False,"use_external":False},"results":{}}
    for det in detection_results:
        results_dict["results"].setdefault(det["sample_token"],[]).append(det)
    with open(det_output/"results_val.json","w") as f: json.dump(results_dict,f,indent=2)
    with open(det_output/"results_test.json","w") as f: json.dump(results_dict,f,indent=2)

    logger.info(f"✅ Fake nuScenes dataset created at {output_dir}")
    logger.info(f"Frames: {len(all_frames)} | Annotations: {len(sample_annotation_json)}")
    logger.info(f"Sanity-check overlays saved in {sanity_dir}")

# ---------------------- CLI ----------------------
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("base_path",type=str)
    p.add_argument("measurement_ids",nargs="+")
    p.add_argument("--output_dir",type=str,default="./data_fake/nuscenes")
    p.add_argument("--max_files",type=int,default=None)
    p.add_argument("--det_output",type=str,default="./data_fake/cp_det")
    args=p.parse_args()
    create_fake_nuscenes(args.base_path,args.measurement_ids,
                         args.output_dir,args.max_files,args.det_output)
