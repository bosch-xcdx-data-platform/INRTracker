import numpy as np
import os
import json
import pickle as pkl
import argparse
import tqdm


def read_json(tracking_result_path):
    with open(tracking_result_path, 'r') as f:
        tracking_result = json.load(f)

    return tracking_result


def read_pkl():
    pass

# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Waymo 3D Detection Evaluation")
#     parser.add_argument("--phase", help="Waymo dataset phase")
#     parser.add_argument(
#         "--work_dir",
#         default="./scripts/data/output_test_pure_det_3dcen",
#         help="the dir which saves the tracking results json file")
#     parser.add_argument("--gt_bin", help="gt.bin file")

#     args = parser.parse_args()

#     return args



if __name__ == '__main__':
    tracking_result_path = os.path.join('/home/julian/workspace/qd-3dt/work_dirs/Waymo/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter_scaled_res_16_0/output_test_pure_det_3dcen', 'output.json')
    tracking_result_path = os.path.join(
        '/home/julian/experiments/tracking/validation_texSDF/Waymo/segment-15096340672898807711_3765_000_3785_000_with_camera_labels', 'outputs.json')
    tracking_result_path = os.path.join(
        '/home/julian/experiments/', 'output.json')
    data_dict = read_json(tracking_result_path=tracking_result_path)