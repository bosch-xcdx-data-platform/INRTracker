#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import tempfile
import pickle

import cv2
import numpy as np
from PIL import Image

from mmengine import Config
from mmdet3d.apis import init_model, inference_mono_3d_detector
from mmdet3d.structures import CameraInstance3DBoxes
from nuscenes.nuscenes import NuScenes

# --- user settings ---
DATA_ROOT = "./data_fake/nuscenes"              # nuScenes root (contains samples/, sweeps/, v1.0-mini/)
VERSION   = "v1.0-mini"                   # or "v1.0-trainval"
CAM       = "CAM_FRONT"
IMG_DIR   = Path(DATA_ROOT) / "samples" / CAM
CFG_FILE = "/home/azureuser/cloudfiles/code/mmdetection3d/configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py"
CKPT_FILE = "./fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth"
OUT_DIR   = Path("fcos3d_out_vis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_intrinsics_for_img(nus, img_path: str):
    rel = os.path.relpath(img_path, DATA_ROOT)  # e.g. samples/CAM_FRONT/xxx.jpg
    tokens = nus.field2token('sample_data', 'filename', rel)
    if not tokens:
        raise RuntimeError(f"No sample_data found for {rel}")
    sd_token = tokens[0]             # take the first (only) match
    sdata = nus.get('sample_data', sd_token)
    cs = nus.get('calibrated_sensor', sdata['calibrated_sensor_token'])
    return cs['camera_intrinsic']    # 3x3 list


def make_temp_ann(img_path: str, K):
    """Minimal annfile required by inference_mono_3d_detector."""
    with Image.open(img_path) as im:
        W, H = im.size
    data_item = {
        "images": {CAM: {"img_path": img_path, "cam2img": K, "ori_shape": (H, W, 3)}},
        "cam2img": K,
    }
    payload = {"metainfo": {"dataset": "nuScenes", "cam_types": [CAM]},
               "data_list": [data_item]}
    fd, pkl_path = tempfile.mkstemp(prefix="mono3d_", suffix=".pkl"); os.close(fd)
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)
    return pkl_path

def draw_cam_box7_on_image(img_bgr, box7, K):
    """box7: [x,y,z,w,h,l,yaw] in CAMERA coords. Draws 12 edges."""
    cb = CameraInstance3DBoxes(np.asarray([box7], dtype=np.float32))
    corners = cb.corners.cpu().numpy()[0]  # (8,3)
    if np.any(corners[:, 2] <= 0.1):
        return False
    K = np.asarray(K, np.float32)
    uvw = corners @ K.T
    z = np.clip(uvw[:, 2:3], 1e-6, None)
    uv = (uvw[:, :2] / z).astype(int)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        cv2.line(img_bgr, tuple(uv[i]), tuple(uv[j]), (0,255,0), 2, cv2.LINE_AA)
    return True

def main():
    # 1) Init model exactly as in the official config (no threshold tweaking)
    cfg = Config.fromfile(CFG_FILE)
    device = "cuda:0" if (os.getenv("CUDA_VISIBLE_DEVICES") or "").strip() != "" else "cpu"
    model = init_model(cfg, CKPT_FILE, device=device)

    # 2) NuScenes API (for intrinsics + file mapping)
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=False)

    # 3) Iterate images, run inference, visualize
    imgs = sorted([*IMG_DIR.glob("*.jpg"), *IMG_DIR.glob("*.png")])
    if not imgs:
        print(f"❌ No images found in {IMG_DIR}")
        return

    for img_path in imgs:
        img_path = str(img_path)
        K = get_intrinsics_for_img(nusc, img_path)          # per-image K
        ann_file = make_temp_ann(img_path, K)               # tiny temp annfile

        result = inference_mono_3d_detector(model, img_path, ann_file, cam_type=CAM)
        # Normalize to InstanceData path
        try:
            pi = result.pred_instances_3d
            boxes = pi.bboxes_3d.tensor.detach().cpu().numpy()  # (N,7): x,y,z,w,h,l,yaw
            scores = pi.scores_3d.detach().cpu().numpy()
        except Exception:
            print(f"(no preds) {os.path.basename(img_path)}")
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        vis = img_bgr.copy()

        # Draw top-50 by score (no custom NMS)
        keep = np.argsort(-scores)[:50]
        for i in keep:
            x, y, z, w, h, l, yaw = boxes[i][:7].tolist()
            draw_cam_box7_on_image(vis, [x,y,z,w,h,l,yaw], K)

        out_png = OUT_DIR / (Path(img_path).stem + ".png")
        cv2.imwrite(str(out_png), vis)
        print(f"✅ Saved {out_png}")

if __name__ == "__main__":
    main()
