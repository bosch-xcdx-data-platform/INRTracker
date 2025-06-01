import os
import argparse
import json
from moviepy.editor import ImageSequenceClip

# Function to create a video from a list of image paths
def create_video(image_paths, output_path, fps=12.0):
    # if output_path exists, delete it
    if os.path.exists(output_path):
        print("Removing existing video: {}".format(output_path))
        os.remove(output_path)
    
    # Sort the image paths based on frame index
    image_paths.sort(key=lambda x: int(x.split("/")[-2].split("_")[-1]))
    
    clip = ImageSequenceClip(image_paths, fps=2)
    clip.write_videofile(output_path)

# Function to process a scene directory
def process_scene(scene_path, cam_id=0):
    rgb_im_w_bbox_paths = []
    synth_im_bbox_paths = []
    
    scene_name = scene_path.split("/")[-1]
    scene_cam_path = os.path.join(scene_path,"vis",f"cam_{cam_id}") # vis/cam_0
    print("Scene Cam Path: {}".format(scene_cam_path))
    

    # Get paths for im_w_bbox.png and im_w_bbox_rgb_out.png images
    for frame_dir in sorted(os.listdir(scene_cam_path)):
        
        frame_path = os.path.join(scene_cam_path, frame_dir)
        if not os.path.isdir(frame_path):
            continue
        
        # last_opt_step = sorted(os.listdir(frame_path))[-1]
        
        rgb_im_w_bbox = os.path.join(frame_path, "im_w_bbox.png")
        synth_im_bbox = os.path.join(frame_path, "im_overlay_bbox.png")

        if os.path.exists(rgb_im_w_bbox) and os.path.exists(synth_im_bbox):
            rgb_im_w_bbox_paths.append(rgb_im_w_bbox)
            synth_im_bbox_paths.append(synth_im_bbox)

    # Create videos for im_w_bbox.png and im_w_bbox_rgb_out.png
    create_video(rgb_im_w_bbox_paths, os.path.join(scene_cam_path, f"{scene_name}_rgb_w_bbox_video.mp4"), fps=2.0)
    create_video(synth_im_bbox_paths, os.path.join(scene_cam_path, f"{scene_name}_rendered_bbox_overlay_video.mp4"), fps=2.0)
    
    if cam_id == 0:
        # Create a video for the entire scene
        create_video(rgb_im_w_bbox_paths, os.path.join(scene_path,"vis", f"{scene_name}_rgb_w_bbox_video_cam0.mp4"), fps=2.0)
        create_video(synth_im_bbox_paths, os.path.join(scene_path,"vis", f"{scene_name}_rendered_bbox_overlay_video_cam0.mp4"), fps=2.0)


def main(args, config):
    data_kwargs = config["data_kwargs"]
    G_kwargs = config["G_kwargs"]
    
    dataset = data_kwargs['dataset']
    split = data_kwargs['split']
    gen_model = G_kwargs['name']
    
    base_path= os.path.join(args.save_dir, dataset, split + '_' + gen_model)
    
    all_scenes = os.listdir(base_path)
    
    for scene in all_scenes:
        scene_viz_path = os.path.join(base_path, scene)
        n_cams = len(os.listdir(scene_viz_path + "/vis"))
        for cam_id in range(n_cams):
            process_scene(scene_viz_path, cam_id)
    
    
if __name__ == "__main__":
    # Params
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default='./configs/configs_track/nuscenes_viz.json')
    parser.add_argument("-sd", "--save_dir", default='./nuscenes_viz')
    
    args, others = parser.parse_known_args()
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    main(args,config)
