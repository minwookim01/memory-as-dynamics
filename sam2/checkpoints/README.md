# Checkpoints

Place model weights here (this directory is git-ignored):

- `sam2.1_hiera_base_plus.pt` — base SAM 2.1 weights used to initialize training.
  Download from the official [SAM 2](https://github.com/facebookresearch/sam2)
  release.
- `rpm_hiera_b+.pt` — RPM checkpoint produced by training (or your released
  weights), used by the inference scripts.

Training reads the base SAM 2.1 checkpoint via
`configs/sam2.1_training/sam2.1_hiera_b+_RPM_train.yaml`
(`checkpoint_path: ./checkpoints/sam2.1_hiera_base_plus.pt`).
