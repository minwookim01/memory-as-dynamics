"""Local (no-SLURM, no-ffmpeg) SA-V preprocessing for RPM / SAM2 training.

The raw SA-V download is laid out as chunks of mp4 videos + json annotations:

    <sav-vid-dir>/sav_000/sav_000889.mp4
    <sav-vid-dir>/sav_000/sav_000889_manual.json
    ...

The training data loader (JSONRawDataset) instead expects extracted JPEG frames
and flat annotation files:

    <out-img-dir>/sav_000889/00000.jpg, 00004.jpg, ...
    <out-gt-dir>/sav_000889_manual.json

This script produces that layout locally with OpenCV (no ffmpeg needed). Point
the training config at:

    img_folder: <out-img-dir>
    gt_folder:  <out-gt-dir>

Example:
    python training/scripts/sav_extract_local.py \
        --sav-vid-dir /data1/data/sa-v/sa-v/sav_train \
        --out-img-dir /data1/data/sa-v/sav_train_frames \
        --out-gt-dir  /data1/data/sa-v/sav_train_jsons \
        --sample-rate 4 --workers 16

Notes:
- Only videos that have a *_manual.json are processed (SA-V also ships auto-only
  videos, which RPM training does not use).
- --sample-rate 4 matches `ann_every: 4`. Frames are named by their true index
  (00000, 00004, ...). Only the kept frames are decoded (grab/retrieve), so 3/4
  of frames are skipped without decoding -> faster.
- Work is streamed (a sliding window of ~2x workers), so the first results print
  within seconds instead of after queuing all videos.
- --workers = number of videos decoded in parallel. Already-extracted videos are
  skipped (resumable).
"""

import argparse
import os
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import cv2


def extract_one(mp4_path, out_img_dir, out_gt_dir, sample_rate, copy_json):
    stem = Path(mp4_path).stem  # e.g. sav_000889
    src_json = Path(mp4_path).with_name(stem + "_manual.json")
    if not src_json.exists():
        return stem, "no_manual_json"

    vid_out = os.path.join(out_img_dir, stem)
    # Resume: skip if frames are already extracted (no marker file left behind,
    # so the frame folder only ever contains %05d.jpg files).
    if os.path.isdir(vid_out) and any(f.endswith(".jpg") for f in os.listdir(vid_out)):
        return stem, "skip"
    os.makedirs(vid_out, exist_ok=True)

    # Keep every sample_rate-th frame. grab() advances without decoding (cheap);
    # retrieve() decodes only the frames we keep -> ~sample_rate times faster.
    cap = cv2.VideoCapture(str(mp4_path))
    fid = 0
    while True:
        if not cap.grab():
            break
        if fid % sample_rate == 0:
            ok, frame = cap.retrieve()
            if ok:
                cv2.imwrite(os.path.join(vid_out, f"{fid:05d}.jpg"), frame)
        fid += 1
    cap.release()

    dst_json = os.path.join(out_gt_dir, stem + "_manual.json")
    if not os.path.exists(dst_json):
        if copy_json:
            shutil.copyfile(src_json, dst_json)
        else:
            os.symlink(os.path.abspath(src_json), dst_json)

    return stem, "ok"


def main():
    parser = argparse.ArgumentParser(description="Local SA-V frame extraction + json flattening (OpenCV)")
    parser.add_argument("--sav-vid-dir", required=True, help="Raw SA-V dir (contains sav_XXX/ chunks)")
    parser.add_argument("--out-img-dir", required=True, help="Output dir for extracted frames (img_folder)")
    parser.add_argument("--out-gt-dir", required=True, help="Output dir for flat manual jsons (gt_folder)")
    parser.add_argument("--sample-rate", type=int, default=4, help="Keep every Nth frame (match ann_every)")
    parser.add_argument("--workers", type=int, default=16, help="Videos decoded in parallel")
    parser.add_argument("--copy-json", action="store_true", help="Copy jsons instead of symlinking")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N videos (debug)")
    args = parser.parse_args()

    os.makedirs(args.out_img_dir, exist_ok=True)
    os.makedirs(args.out_gt_dir, exist_ok=True)

    mp4_files = sorted(str(p) for p in Path(args.sav_vid_dir).glob("*/*.mp4"))
    if args.limit:
        mp4_files = mp4_files[: args.limit]
    total = len(mp4_files)
    print(f"Found {total} mp4 videos | {args.workers} workers", flush=True)
    print("Starting extraction...", flush=True)

    def submit(ex, it):
        m = next(it, None)
        return ex.submit(extract_one, m, args.out_img_dir, args.out_gt_dir, args.sample_rate, args.copy_json) if m else None

    it = iter(mp4_files)
    done = 0
    skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        # Prime a sliding window of ~2x workers so results start flowing at once.
        inflight = set()
        for _ in range(args.workers * 2):
            fut = submit(ex, it)
            if fut is None:
                break
            inflight.add(fut)

        while inflight:
            finished, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            inflight = set(inflight)
            for fut in finished:
                stem, status = fut.result()
                done += 1
                if status == "no_manual_json":
                    skipped += 1
                print(f"[{done}/{total}] ({100 * done / total:.1f}%) {stem} {status}", flush=True)
                nxt = submit(ex, it)
                if nxt is not None:
                    inflight.add(nxt)

    print(f"Done. frames -> {args.out_img_dir} | jsons -> {args.out_gt_dir}")
    if skipped:
        print(f"Skipped {skipped} videos without a *_manual.json")


if __name__ == "__main__":
    main()
