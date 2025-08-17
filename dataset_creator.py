#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, argparse
from pathlib import Path
import cv2, numpy as np
from uuid import uuid4
from datetime import datetime

# ===== Hardcoded Extrinsics for "front" camera (ego->camera) =====
FRONT_EXTRINSIC = {
    "x": 3.869674600444828,
    "y": 0.07390659625957596,
    "z": 0.6864303987155654,
    "roll": -179.952,  # deg
    "pitch": 16.4037,  # deg
    "yaw": 1.12799     # deg
}

# ===== Intrinsics from XML (approx. pinhole) =====
FX = 558.2733764648438 * 0.9997516870498657
FY = 558.2733764648438
CX = 959.7113037109375
CY = 768.0039672851562
CAM_INTRINSIC = np.array([
    [FX, 0.0, CX],
    [0.0, FY, CY],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# ------------------ math utils ------------------

def rpy_to_matrix(roll, pitch, yaw):
    r, p, y = np.deg2rad([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0],[0, np.cos(r), -np.sin(r)],[0, np.sin(r),  np.cos(r)]], np.float64)
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],[0, 1, 0],[-np.sin(p), 0, np.cos(p)]], np.float64)
    Rz = np.array([[np.cos(y), -np.sin(y), 0],[np.sin(y),  np.cos(y), 0],[0, 0, 1]], np.float64)
    return Rz @ Ry @ Rx  # Z * Y * X

def rotmat_to_quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            w = (R[2,1] - R[1,2]) / s; x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s; z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s; y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s; y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
    return [float(w), float(x), float(y), float(z)]

def extrinsic_matrix_from_params(params):
    R = rpy_to_matrix(params["roll"], params["pitch"], params["yaw"])
    t = np.array([params["x"], params["y"], params["z"]], np.float64).reshape((3,1))
    T = np.eye(4, dtype=np.float64); T[:3,:3] = R; T[:3,3:] = t
    return T

# ------------------ image geometry ------------------

def rotate_intrinsics_180(K, width, height):
    K2 = K.copy()
    K2[0, 2] = (width - 1) - K2[0, 2]
    K2[1, 2] = (height - 1) - K2[1, 2]
    return K2

def crop_center_resize(frame, target_w=1600, target_h=900):
    """(Old path) Center-crop to 16:9, then resize."""
    h, w = frame.shape[:2]
    target_aspect = target_w / target_h
    input_aspect = w / h
    if input_aspect > target_aspect:
        new_w = int(round(h * target_aspect)); x0 = (w - new_w) // 2
        cropped = frame[:, x0:x0+new_w]
        sx = target_w / new_w; sy = target_h / h
        cx_shift = -x0; cy_shift = 0
    else:
        new_h = int(round(w / target_aspect)); y0 = (h - new_h) // 2
        cropped = frame[y0:y0+new_h, :]
        sx = target_w / w; sy = target_h / new_h
        cx_shift = 0; cy_shift = -y0
    out = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return out, sx, sy, cx_shift, cy_shift

def letterbox_resize(frame, target_w=1600, target_h=900):
    """
    Uniform scale to fit within target, then pad with black borders (letterbox/pillarbox).
    Returns: out_img, scale, dx, dy where dx,dy are left/top padding in pixels.
    Intrinsics update: fx,fy *= scale; cx = cx*scale + dx; cy = cy*scale + dy
    """
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)  # black borders
    dx = (target_w - new_w) // 2
    dy = (target_h - new_h) // 2
    out[dy:dy+new_h, dx:dx+new_w] = resized
    return out, scale, dx, dy

# ------------------ IO helpers ------------------

def ensure_blank_map_png(maps_dir: Path) -> str:
    maps_dir.mkdir(parents=True, exist_ok=True)
    blank_path = maps_dir / "blank.png"
    if not blank_path.exists():
        img = np.full((512, 512, 3), 255, dtype=np.uint8)
        cv2.putText(img, "FAKE MAP", (160, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
        cv2.imwrite(str(blank_path), img)
    return "maps/blank.png"

# ------------------ main ------------------

def create_fake_nuscenes(mp4_path, output_dir="./data_fake/nuscenes",
                         target_w=1600, target_h=900,
                         fit="letterbox",   # "letterbox" or "crop"
                         rotate_180_pixels=True):
    output_dir = Path(output_dir)
    samples_dir = output_dir / "samples" / "CAM_FRONT"
    maps_dir = output_dir / "maps"
    version_dir = output_dir / "v1.0-mini"
    samples_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)
    version_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mp4_path))
    frame_id = 0
    frame_files = []
    K_final = None
    W = target_w; H = target_h

    base_ts_us = int(datetime.utcnow().timestamp() * 1e6)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if rotate_180_pixels:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        if fit == "letterbox":
            # keep full FOV, add black borders
            frame_out, scale, dx, dy = letterbox_resize(frame, target_w, target_h)
            if K_final is None:
                K1 = CAM_INTRINSIC.copy()
                K1[0,0] *= scale; K1[1,1] *= scale
                K1[0,2] = K1[0,2] * scale + dx
                K1[1,2] = K1[1,2] * scale + dy
                if rotate_180_pixels:
                    K1 = rotate_intrinsics_180(K1, target_w, target_h)
                K_final = K1
        elif fit == "crop":
            # old behavior
            frame_out, sx, sy, cx_shift, cy_shift = crop_center_resize(frame, target_w, target_h)
            if K_final is None:
                K1 = CAM_INTRINSIC.copy()
                K1[0, 2] = (K1[0, 2] + cx_shift) * sx
                K1[1, 2] = (K1[1, 2] + cy_shift) * sy
                K1[0, 0] = K1[0, 0] * sx
                K1[1, 1] = K1[1, 1] * sy
                if rotate_180_pixels:
                    K1 = rotate_intrinsics_180(K1, target_w, target_h)
                K_final = K1
        else:
            raise ValueError("--fit must be 'letterbox' or 'crop'")

        # write image
        fname = f"{frame_id:06d}.jpg"
        cv2.imwrite(str(samples_dir / fname), frame_out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_files.append((fname, base_ts_us + frame_id * 500000))
        frame_id += 1

    cap.release()
    if not frame_files:
        raise RuntimeError("No frames extracted from the video.")

    # --- nuScenes-like tables ---
    SENSOR_TOKEN = str(uuid4()); CALIB_TOKEN = str(uuid4()); LOG_TOKEN = str(uuid4())
    SCENE_TOKEN = str(uuid4()); MAP_TOKEN = str(uuid4())

    map_filename_rel = ensure_blank_map_png(maps_dir)
    map_json = [{"token": MAP_TOKEN, "log_tokens": [LOG_TOKEN], "category": "semantic_prior", "filename": map_filename_rel}]
    log_json = [{"token": LOG_TOKEN, "logfile": "fake_log", "vehicle": "fake_vehicle", "date_captured": "2025-01-01", "location": "singapore-onenorth"}]
    sensor_json = [{"token": SENSOR_TOKEN, "channel": "CAM_FRONT", "modality": "camera"}]

    T_ego_cam = extrinsic_matrix_from_params(FRONT_EXTRINSIC)  # do NOT add 180° here
    R_ego_cam = T_ego_cam[:3,:3]; t_ego_cam = T_ego_cam[:3,3].tolist()
    q_ego_cam = rotmat_to_quat_wxyz(R_ego_cam)
    calibrated_sensor_json = [{
        "token": CALIB_TOKEN,
        "sensor_token": SENSOR_TOKEN,
        "translation": [float(t) for t in t_ego_cam],
        "rotation": q_ego_cam,
        "camera_intrinsic": K_final.tolist()
    }]

    ego_pose_json = []
    for _, ts in frame_files:
        ego_pose_json.append({"token": str(uuid4()), "timestamp": int(ts),
                              "translation": [0.0,0.0,0.0], "rotation": [1.0,0.0,0.0,0.0]})

    sample_json = []; sample_data_json = []
    first_tok = None; last_tok = None
    for i,(fname,ts) in enumerate(frame_files):
        smp_tok = str(uuid4()); prev = sample_json[-1]["token"] if sample_json else ""
        if first_tok is None: first_tok = smp_tok
        last_tok = smp_tok
        sample_json.append({"token": smp_tok, "timestamp": int(ts), "scene_token": SCENE_TOKEN,
                            "prev": prev, "next":"", "data": {}})
        if prev: sample_json[-2]["next"] = smp_tok
        sd_tok = str(uuid4()); ego_pose_tok = ego_pose_json[i]["token"]
        sample_data_json.append({
            "token": sd_tok, "sample_token": smp_tok, "ego_pose_token": ego_pose_tok,
            "calibrated_sensor_token": CALIB_TOKEN, "timestamp": int(ts),
            "fileformat": "jpg", "is_key_frame": True, "height": int(H), "width": int(W),
            "filename": f"samples/CAM_FRONT/{fname}"
        })
        sample_json[-1]["data"]["CAM_FRONT"] = sd_tok

    scene_json = [{
        "token": SCENE_TOKEN, "name": "scene-0001",
        "description": f"Fake scene from MP4 with {fit} to {W}x{H} and 180° pixel rotation (intrinsics compensated).",
        "nbr_samples": len(frame_files), "first_sample_token": first_tok,
        "last_sample_token": last_tok, "log_token": LOG_TOKEN
    }]

    for name, content in {
        "scene.json": scene_json, "sample.json": sample_json, "sample_data.json": sample_data_json,
        "ego_pose.json": ego_pose_json, "sensor.json": sensor_json,
        "calibrated_sensor.json": calibrated_sensor_json, "log.json": log_json,
        "map.json": map_json, "attribute.json": [], "category.json": [],
        "instance.json": [], "sample_annotation.json": [], "visibility.json": []
    }.items():
        with open(version_dir / name, "w") as f: json.dump(content, f, indent=2)

    intrinsics_path = Path("./cam_front_intrinsics.json")
    with open(intrinsics_path, "w") as f:
        json.dump({
            "fx": float(K_final[0,0]), "fy": float(K_final[1,1]),
            "cx": float(K_final[0,2]), "cy": float(K_final[1,2]),
            "width": int(W), "height": int(H),
            "fit": fit, "rotated_180": bool(rotate_180_pixels)
        }, f, indent=2)

    print(f"✅ Fake nuScenes dataset created at: {output_dir}")
    print(f"   Frames: {len(frame_files)} | Image size: {W}x{H}")
    print("   Intrinsics after resize+padding and 180° pixel rotation (if enabled):")
    print(f"     fx, fy, cx, cy = {K_final[0,0]:.6f} {K_final[1,1]:.6f} {K_final[0,2]:.6f} {K_final[1,2]:.6f}")

# ------------------ CLI ------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Create fake nuScenes dataset from MP4 with letterbox/crop to 1600x900, optional 180° pixel rotate, and consistent v1.0-mini tables."
    )
    p.add_argument("--mp4_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./data_fake/nuscenes")
    p.add_argument("--fit", type=str, choices=["letterbox","crop"], default="letterbox",
                   help="letterbox keeps full FOV (black borders); crop trims FOV to 16:9")
    p.add_argument("--no_rotate180", action="store_true", help="disable 180° pixel rotation")
    args = p.parse_args()
    create_fake_nuscenes(args.mp4_path, args.output_dir, 1600, 900, args.fit, not args.no_rotate180)
