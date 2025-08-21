import os
import sys
import argparse
import pickle
import numpy as np
from nuscenes.nuscenes import NuScenes
sys.path.append('./yolo12')
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_root', type=str, default='./data/nuscenes', help='Path to the NuScenes dataset root directory')
    parser.add_argument('--version', type=str, default='v1.0-trainval-select', help='NuScenes dataset version')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--info_path', type=str, default='./data/nuscenes/nuscenes_data_info.pkl', help='Path to the NuScenes data info file')
    parser.add_argument('--model_ckpt', type=str, default='./yolo12/ckpts/yolov12l-seg.engine', help='Path to the YOLO model checkpoint')
    parser.add_argument('--save_dir', type=str, default='./data/nuscenes/img_results', help='Directory to save the image processing results')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=args.verbose)
    with open(args.info_path, 'rb') as f:
        data_info = pickle.load(f)
    classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    cams = data_info[0]['samples'][0]['cams'].keys()
    model = YOLO(args.model_ckpt)
    
    for scene in data_info:
        scene_name = scene['scene_name']
        scene_save_dir = f"{args.save_dir}/{scene_name}"
        cam_tokens = {cam: scene['samples'][0]['cams'][cam]['token'] for cam in cams}
        os.makedirs(scene_save_dir, exist_ok=True)

        for cam in cams:
            os.makedirs(os.path.join(scene_save_dir, cam), exist_ok=True)

            for i in range(len(scene['samples'])):
                sample = scene['samples'][i]
                now_cam_token = sample['cams'][cam]['token']
                flag = False

                while True:
                    if cam_tokens[cam] == '':
                        cam_info = nusc.get('sample_data', now_cam_token[cam])
                        while cam_info['prev'] != '':
                            cam_info = nusc.get('sample_data', cam_info['prev'])
                        cam_tokens[cam] = cam_info['token']
                    if now_cam_token == cam_tokens[cam]:
                        flag = True
                    cam_info = nusc.get('sample_data', cam_tokens[cam])
                    img_path = os.path.join(args.data_root, cam_info['filename'])
                    cam_tokens[cam] = cam_info['next']
                    results = model.track(img_path, imgsz=1280, classes=list(classes.keys()), persist=True, retina_masks=True, verbose=False)
                    if flag:
                        save_results = {
                            'bboxes': results[0].boxes.xywh.cpu().numpy() if results[0].boxes.xywh is not None else None,
                            'labels': results[0].boxes.cls.cpu().numpy().astype(int) if results[0].boxes.cls is not None else None,
                            'ids': results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else None,
                            'scores': results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else None,
                            'masks': [mask.astype(int) for mask in results[0].masks.xy] if results[0].masks is not None else None,
                        }
                        save_path = os.path.join(scene_save_dir, cam, os.path.basename(img_path).replace('.jpg', '.pkl'))
                        with open(save_path, 'wb') as f:
                            pickle.dump(save_results, f)
                        break

            model.predictor.trackers[0].reset()

        print(f"Processed scene: {scene_name} with {len(scene['samples'])} samples.")