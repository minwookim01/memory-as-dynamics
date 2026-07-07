"""Run RPM inference on the LaSOT test set, split into chunks.

Launch one process per chunk to parallelize across GPUs/processes, e.g.:
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$i python scripts/main_inference_chunk_lasot.py \
            --data_root /path/to/LaSOT \
            --checkpoint checkpoints/rpm_hiera_b+.pt \
            --output results/lasot --chunk_idx $i --num_chunks 4 &
    done
"""

import argparse
import gc
import os
import os.path as osp
import time

import numpy as np
import torch
import tqdm

from sam2.build_sam import build_sam2_video_predictor


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
    parser = argparse.ArgumentParser(description="RPM chunked inference on LaSOT")
    parser.add_argument("--data_root", required=True,
                        help="LaSOT root containing <video>/img and groundtruth.txt")
    parser.add_argument("--testing_set", default=None,
                        help="testing_set.txt (default: <data_root>/testing_set.txt)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/rpm/lasot/sam2.1_hiera_b+.yaml")
    parser.add_argument("--output", default="results/lasot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    testing_set = args.testing_set or osp.join(args.data_root, "testing_set.txt")
    os.makedirs(args.output, exist_ok=True)

    with open(testing_set, "r") as f:
        all_videos = [v.strip() for v in f.readlines() if v.strip()]

    per_chunk = (len(all_videos) + args.num_chunks - 1) // args.num_chunks
    start = args.chunk_idx * per_chunk
    end = min(start + per_chunk, len(all_videos))
    chunk_videos = all_videos[start:end]
    print(f"[chunk {args.chunk_idx}/{args.num_chunks}] videos {start}..{end - 1} ({len(chunk_videos)})")

    for local_vid, video in enumerate(chunk_videos):
        frame_folder = osp.join(args.data_root, video, "img")
        num_frames = len(os.listdir(frame_folder))
        print(f"[chunk {args.chunk_idx}] [{local_vid + 1}/{len(chunk_videos)}] {video} ({num_frames} frames)")

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
                mask = masks[0][0].cpu().numpy() > 0
                non_zero = np.argwhere(mask)
                if len(non_zero) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero.min(axis=0)
                    y_max, x_max = non_zero.max(axis=0)
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                predictions.append(bbox)

        with open(osp.join(args.output, f"{video}.txt"), "w") as f:
            for x, y, w, h in predictions:
                f.write(f"{x},{y},{w},{h}\n")

        del predictor, state
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

    print(f"[chunk {args.chunk_idx}] done")


if __name__ == "__main__":
    main()
