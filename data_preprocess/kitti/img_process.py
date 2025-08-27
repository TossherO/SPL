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
    parser.add_argument('--data_root', type=str, default='./data/kitti', help='Path to the KITTI dataset root directory')
    parser.add_argument('--info_path', type=str, default='./data/kitti/kitti_data_info.pkl', help='Path to the KITTI data info file')
    parser.add_argument('--model_ckpt', type=str, default='./yolo12/ckpts/yolov12l-seg.engine', help='Path to the YOLO model checkpoint')
    parser.add_argument('--save_dir', type=str, default='./data/kitti/img_results', help='Directory to save the image processing results')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    with open(args.info_path, 'rb') as f:
        data_info = pickle.load(f)
    classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle'}
    model = YOLO(args.model_ckpt)
    
    for scene in data_info['kitti_raw'].keys():
        scene_save_dir = f"{args.save_dir}/{scene}"
        os.makedirs(scene_save_dir, exist_ok=True)
        frame_names = list(data_info['kitti_raw'][scene].keys())
        frame_names.sort()

        last_frame_idx = -1
        for name in frame_names:
            frame_idx = int(name)
            if last_frame_idx != -1 and frame_idx != last_frame_idx + 1:
                model.predictor.trackers[0].reset()

            img_path = data_info['kitti_raw'][scene][name]['img_path']
            results = model.track(img_path, imgsz=1024, classes=list(classes.keys()), persist=True, retina_masks=True, verbose=False)
            save_results = {
                'bboxes': results[0].boxes.xywh.cpu().numpy() if results[0].boxes.xywh is not None else None,
                'labels': results[0].boxes.cls.cpu().numpy().astype(int) if results[0].boxes.cls is not None else None,
                'ids': results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else None,
                'scores': results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else None,
                'masks': [mask.astype(int) for mask in results[0].masks.xy] if results[0].masks is not None else None,
            }
            last_frame_idx = frame_idx
            save_path = os.path.join(scene_save_dir, name + '.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(save_results, f)

        model.predictor.trackers[0].reset()
        print(f"Processed scene: {scene} with {len(frame_names)} frames.")