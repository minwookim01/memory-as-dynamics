"""Run RPM inference on the LaSOT test set (one video at a time).

Example:
    python scripts/main_inference_lasot.py \
        --data_root /path/to/LaSOT \
        --checkpoint checkpoints/rpm_hiera_b+.pt \
        --output results/lasot

Predictions are written as ``<video>.txt`` (one ``x,y,w,h`` box per frame),
matching the LaSOT evaluation format.
"""

import argparse
import gc
import os
import os.path as osp
import time

import cv2
import numpy as np
import torch
import tqdm
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor


def load_mask(mask_path):
    mask = np.asarray(Image.open(mask_path)).astype(np.float32)
    return (mask > 0).astype(np.uint8)


def load_lasot_gt(gt_path):
    """Read a LaSOT groundtruth file; the first-frame box is used as the prompt."""
    with open(gt_path, "r") as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(int, line.split(","))
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts


def parse_args():
    parser = argparse.ArgumentParser(description="RPM inference on LaSOT")
    parser.add_argument("--data_root", required=True,
                        help="LaSOT root containing <video>/img and groundtruth.txt")
    parser.add_argument("--testing_set", default=None,
                        help="testing_set.txt listing video names (default: <data_root>/testing_set.txt)")
    parser.add_argument("--checkpoint", required=True, help="Path to the RPM checkpoint (.pt)")
    parser.add_argument("--config", default="configs/rpm/lasot/sam2.1_hiera_b+.yaml",
                        help="Model config (Hydra path relative to the sam2 package)")
    parser.add_argument("--output", default="results/lasot", help="Directory for prediction txt files")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    testing_set = args.testing_set or osp.join(args.data_root, "testing_set.txt")
    os.makedirs(args.output, exist_ok=True)

    with open(testing_set, "r") as f:
        test_videos = [v.strip() for v in f.readlines() if v.strip()]

    print(f"Running RPM on {len(test_videos)} LaSOT videos ({args.device})")

    for vid, video in enumerate(test_videos):
        frame_folder = osp.join(args.data_root, video, "img")
        num_frames = len(os.listdir(frame_folder))
        print(f"[{vid + 1}/{len(test_videos)}] {video} ({num_frames} frames)")

        predictor = build_sam2_video_predictor(args.config, args.checkpoint, device=args.device)
        predictions = []

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = predictor.init_state(
                frame_folder,
                offload_video_to_cpu=False,
                offload_state_to_cpu=False,
                async_loading_frames=True,
            )

            prompts = load_lasot_gt(osp.join(args.data_root, video, "groundtruth.txt"))
            bbox, _ = prompts[0]
            predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

            for frame_idx, object_ids, masks in tqdm.tqdm(predictor.propagate_in_video(state)):
                assert len(masks) == 1 and len(object_ids) == 1, "Only single-object tracking is supported"
                mask = masks[0][0].cpu().numpy() > 0.0
                non_zero = np.argwhere(mask)
                if len(non_zero) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero.min(axis=0).tolist()
                    y_max, x_max = non_zero.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                predictions.append(bbox)

        with open(osp.join(args.output, f"{video}.txt"), "w") as f:
            for x, y, w, h in predictions:
                f.write(f"{x},{y},{w},{h}\n")

        # Release GPU/CPU state between videos.
        del predictor, state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()
        time.sleep(1)


if __name__ == "__main__":
    main()
