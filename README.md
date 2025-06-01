# Towards Generalizable and Interpretable Vision with Inverse Neural Rendering

We provide supplemental code to implement and run the multi-object tracking method presented in *Towards Generalizable and Interpretable Vision with Inverse Neural Rendering*. Following the steps below should produce qualitative result videos for all validation scenes (scene-0103, scene-0553, scene-0796, scene-916) in **nuScenes Mini**, a subset of 10 scenes from the official trainval split of the nuScenes dataset.

To run all **other scenes** and **cameras** in the validation or test set, please change `"split": "mini"` under `"data_kwargs"` in `configs/configs_track/nuscenes_viz.json` to `"split": "validation"` or `"split": "testing"`. 

## 0. Requirements
- This code has only been tested on Linux
- A single NVIDIA GPU of the Ampere or any later generation
- Compatability with CUDA toolkit 11.3 or later

## 1. Setup Environment and Install Requirements

To install all required submodules (nuScenes devkit, GET3D) run:

```bash
cd submodules
git clone git@github.com:nv-tlabs/GET3D.git
git clone git@github.com:nutonomy/nuscenes-devkit.git
cd ../
```

Pytorch3D requires this specific order of instructions to run without issues. To setup a conda environment run:
```bash
conda create -y -n inr python=3.9
conda activate inr
conda install  -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install  -y iopath nvidiacub jupyter fvcore -c fvcore -c conda-forge -c iopath -c bottler 
pip install scikit-image matplotlib imageio plotly opencv-python black usort flake8 flake8-bugbear flake8-comprehensions 
conda install -y pytorch3d -c pytorch3d

pip install filterpy==1.4.5 tqdm trimesh==3.23.5 nuscenes-devkit lpips ninja xatlas gdown urllib3 waymo-open-dataset-tf-2-11-0 moviepy
mkdir data
ln -s ../submodules/GET3D/data/tets data/

pip install git+https://github.com/NVlabs/nvdiffrast/
```

Make sure to downgrade `mkl` package to version `2024.0.0` as follows.
```bash
pip uninstall mkl
pip install mkl==2024.0.0
```

## 2. Dataset and Checkpoint Preperation

### nuScenes Mini
- Create an account or sign in on [nuScenes Download](https://www.nuscenes.org/nuscenes#download) 
- Download **Mini** from the **Full dataset (v1.0)** section. Unzip the downloaded zip file and copy all files and folders in the unzipped directory `v1.0-mini` to `./data/nuscenes`.
- Download **Map expansion pack v1.3** from the **Map expansion** section. Unzip the downloaded zip file and copy all files and folders in the unzipped directory `maps` to `./data/nuscenes`.

The structure of the `data` directory should be as follows.
```
${ROOT}
|-- data
`-- |-- nuscenes
    `-- |-- maps
        |-- samples
        |   |-- CAM_BACK
        |   |   | -- xxx.jpg
        |   |-- CAM_BACK_LEFT
        |   |-- CAM_BACK_RIGHT
        |   |-- CAM_FRONT
        |   |-- CAM_FRONT_LEFT
        |   `-- CAM_FRONT_RIGHT
        |-- sweeps
        |   |-- CAM_BACK
        |   |-- CAM_BACK_LEFT
        |   |-- CAM_BACK_RIGHT
        |   |-- CAM_FRONT
        |   |-- CAM_FRONT_LEFT
        |   `-- CAM_FRONT_RIGHT
        |-- v1.0-mini
```

### CenterPoint Detections
Download and extract detections from CenterPoint for the nuscesn validation and test set from [cp_det.zip](https://light.princeton.edu/wp-content/uploads/2024/08/cp_det.zip):

```bash
cd data/
wget https://light.princeton.edu/wp-content/uploads/2024/08/cp_det.zip
unzip cp_det.zip
rm cp_det.zip
cd ..
```

### Get3D Checkpoint
Please download shapenet_car.pt from the [GET3D Checkpoint Release](https://huggingface.co/JunGaoNVIDIA/get3d-ckpt/blob/main/shapenet_car.pt).
Move it to `./ckpt/shapenet_car.pt`

```bash
cd ckpt/
gdown https://huggingface.co/JunGaoNVIDIA/get3d-ckpt/blob/main/shapenet_car.pt
cd ..
```

## 3. Run Tracker

```bash
export NUSC_DATA_PATH=./data/nuscenes
export CENTERPOINT_DETECTION_PATH=./data/cp_det
export GEN3D_CKPT_PATH=./ckpt/shapenet_car.pt
python run_tracker.py -c configs/configs_track/nuscenes_viz.json -sd nuscenes_viz -rn nuscenes_viz --generator_checkpoint $GEN3D_CKPT_PATH --tracking_pth $NUSC_DATA_PATH --detections $CENTERPOINT_DETECTION_PATH --viz
```

Argument Descriptions:
```
-c [Config file that specifies all experiment settings.]
-sd  [Directory where the experiments are saved]
-rn  [Experiment name to run like 'mushrooms' (the rest of the experiment groups are in exp_configs/sps_exps.py)] 
--generator_checkpoint [Checkpoint file of the 3D generation method]
--tracking_pth [Path to the autonomous driving dataset]
--detections [Directory where the datasets are saved]
--viz [If included, visualizations are saved.]
```

## 4. Visualize the Results

If the `--viz`-flag is set, visualizations of the tracker and all intermediate optimization steps (similar to **Fig.1 b**) will be saved in `<sd>\nuScenes\mini_GET3D\scene-<scene_name>\vis`. 

Images for one scene are combined and saved to a video `<sd>\nuScenes\mini_GET3D\scene-<scene_name>\vis\<camera_id>\*.mp4` by running the following command:

```bash
python render_video.py -c configs/configs_track/nuscenes_viz.json -sd nuscenes_viz
```

##

Any *"start_idx":i* and *"stop_idx":i+1* should work on the Mini dataset for *i = [0,9]*. The nuScenes **Mini** set **includes training scenes** as well, which use **perfect bounding boxes** from the dataset annotations.
To run other scenes in Mini that are part of the nuScenes **validation** or **test** set and use **CP detections** change start_idx and stop_idx `configs/configs_track/nuscenes_viz.json` to one of the following:

**scene-0103**

"start_idx":1,
"stop_idx":2,

**scene-0553**

"startidx":2,
"stop_idx":3,

**scene-0796**

"start_idx":5,
"stop_idx":6,

**scene-916**

"start_idx":6,
"stop_idx":7,
