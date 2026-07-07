<h1 align="center">Memory as Dynamics: Learning Reliability-Guided Predictive Models for Online Video Perception</h1>

<p align="center">
  <b>Minwoo Kim</b><sup>1</sup>, <b><a href="https://scholar.google.com/citations?user=V6HVW-QAAAAJ&hl=ko&oi=ao">Sang Min Yoon</a></b><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>HCI Lab, College of Computer Science, Kookmin University, Seoul, Korea
</p>

<p align="center">
  <img src="assets/main_figure.png" width="90%">
</p>

---

## Installation

```bash
# 1. Create the environment and install PyTorch (CUDA 12.1)
conda create -n rpm python=3.10 -y
conda activate rpm
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# 2. Install the remaining dependencies
pip install -r requirements.txt

# 3. Install Mamba (predictive-memory dynamics)
pip install packaging ninja setuptools wheel
pip install causal-conv1d==1.4.0 --no-build-isolation --no-deps
pip install mamba-ssm==2.2.2     --no-build-isolation --no-deps

# 4. Install the `sam2` package
cd sam2
pip install -e .
cd ..
```

---

## Data and Checkpoints

### Datasets

Download and lay out the tracking datasets as:

```
LaSOT/
├── testing_set.txt              # list of test video names
└── <video>/
    ├── img/00000001.jpg ...
    └── groundtruth.txt          # x,y,w,h per frame

LaSOT_ext/                       # same layout as LaSOT
```

Training uses video-segmentation data (SA-V, optionally MOSE / VOS-2019). Edit
the paths in `sam2/sam2/configs/sam2.1_training/sam2.1_hiera_b+_RPM_train.yaml`.

### Checkpoints

Weights live under `sam2/checkpoints/`.

Download the base SAM 2.1 weights (needed to initialize training and for the
SAM2 backbone):

```bash
cd sam2/checkpoints
bash download_ckpts.sh
cd ../..
```

| file | purpose |
|------|---------|
| `sam2.1_hiera_base_plus.pt` | base SAM 2.1 weights (from `download_ckpts.sh`), to initialize training |
| `rpm_hiera_b+.pt` | RPM weights produced by training, used for inference |

---

## Training

RPM training **only learns the predictive-memory prompt** — the SAM2 backbone is
frozen. Run from the `sam2/` directory:

```bash
cd sam2
python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_b+_RPM_train.yaml \
    --use-cluster 0 \
    --num-gpus 4
```

- `--num-gpus`: number of GPUs on this node.
- The base SAM 2.1 checkpoint is read from
  `./checkpoints/sam2.1_hiera_base_plus.pt` (set in the config).
- Trained checkpoints are written under the experiment log directory; copy the
  final one to `sam2/checkpoints/rpm_hiera_b+.pt` for inference.

---

## Inference

### LaSOT (single process)

```bash
python scripts/main_inference_lasot.py \
    --data_root /path/to/LaSOT \
    --checkpoint sam2/checkpoints/rpm_hiera_b+.pt \
    --config configs/rpm/lasot/sam2.1_hiera_b+.yaml \
    --output results/lasot
```

### LaSOT / LaSOT-ext (chunked across GPUs)

```bash
# LaSOT, 4 processes / 4 GPUs
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python scripts/main_inference_chunk_lasot.py \
      --data_root /path/to/LaSOT \
      --checkpoint sam2/checkpoints/rpm_hiera_b+.pt \
      --config configs/rpm/lasot/sam2.1_hiera_b+.yaml \
      --output results/lasot --chunk_idx $i --num_chunks 4 &
done
wait
```

`main_inference_chunk_lasot_ext.py` is the LaSOT-ext counterpart (same arguments).
Each script writes one `<video>.txt` per sequence (`x,y,w,h` per frame).

---

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{kim2026rpm,
  title     = {Memory as Dynamics: Learning Reliability-Guided Predictive Models for Online Video Perception},
  author    = {Kim, Minwoo and Yoon, Sang Min},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

---

## Acknowledgements

This codebase is built on [SAM 2](https://github.com/facebookresearch/sam2). The
long-term memory management strategy follows
[HiM2SAM](https://github.com/LouisFinner/HiM2SAM); the predictive-memory dynamics
are implemented with [Mamba](https://github.com/state-spaces/mamba).
