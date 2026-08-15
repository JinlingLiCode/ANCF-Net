# ANCF-Net: Adaptive Neighborhood Context Fusion with Dynamic Scale Modeling for High-Fidelity Point Cloud Upsampling

Official PyTorch implementation of **ANCF-Net: Adaptive Neighborhood Context
Fusion with Dynamic Scale Modeling for High-Fidelity Point Cloud Upsampling**.

The repository contains the training and inference code, pretrained models for
PU-GAN, PU1K, and Sketchfab, and the evaluation scripts used to compute CD,
HD, EMD, P2F-avg, and P2F-std in the manuscript.

## Environment

The experiments were conducted with:

- Ubuntu
- Python 3.7.11
- PyTorch 1.7.1
- CUDA 11.0

One possible installation is:

```bash
conda create -n ancfnet python=3.7.11
conda activate ancfnet
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

Compile the CUDA operators used by ANCF-Net:

```bash
cd models/Chamfer3D
python setup.py install
cd ../pointops
python setup.py install
cd ../..
```

## Data preparation

Download the [PU-GAN](https://github.com/liruihui/PU-GAN) and
[PU1K](https://github.com/guochengqian/PU-GCN) datasets from their official
project pages. Dataset files are not redistributed by this repository.

The default training files are:

```text
data/
├── PU-GAN/
│   ├── train/PUGAN_poisson_256_poisson_1024.h5
│   └── test/                                  # reference OFF meshes
├── PU1K/
│   ├── train/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5
│   └── test/
│       ├── input_2048/{input_2048,gt_8192}/   # official 4x test split
│       └── original_meshes/                    # reference OFF meshes
└── Sketchfab/
    ├── train/Self_sketchfab_256_1024_poisson.h5
    └── test/input_2048/input_2048/            # input XYZ files
```

Generate 2,048-point inputs and the corresponding ground truth from the
PU-GAN test meshes:

```bash
# 4x: 2,048 -> 8,192 points
python prepare_pugan.py --input_pts_num 2048 --gt_pts_num 8192 --seed 21

# 16x: 2,048 -> 32,768 points
python prepare_pugan.py --input_pts_num 2048 --gt_pts_num 32768 --seed 21
```

The official PU1K download already contains the 4x `input_2048` test split.
The same mesh-sampling utility is available when a regenerated 4x or 16x
split is required:

```bash
python prepare_pu1k.py --input_pts_num 2048 --gt_pts_num 8192 --seed 21
python prepare_pu1k.py --input_pts_num 2048 --gt_pts_num 32768 --seed 21
```

Poisson-disk sampling in Open3D can vary across library versions. Use the
reported Open3D version and seed when regenerating the test point clouds.

## Pretrained models and inference

The provided checkpoints are:

```text
pretrained_model/pugan/ckpt/pugan.pth
pretrained_model/pu1k/ckpt/pu1k.pth
pretrained_model/Sketchfab/ckpt/Sketchfab.pth
```

PU-GAN:

```bash
python test.py --dataset pugan --up_rate 4 --save_dir 4X
python test.py --dataset pugan --up_rate 16 --save_dir 16X
```

PU1K:

```bash
python test.py --dataset pu1k --up_rate 4 --save_dir 4X
python test.py --dataset pu1k --up_rate 16 --save_dir 16X
```

Sketchfab:

```bash
python test.py --dataset Sketchfab --up_rate 4 --save_dir 4X
python test.py --dataset Sketchfab --up_rate 16 --save_dir 16X
```

The 16x setting applies the 4x pipeline twice. Use `--test_input_path` and
`--ckpt_path` to override the default paths. Predictions are written to the
`test/<save_dir>` directory next to the corresponding pretrained-model
folder. For example, PU-GAN 4x predictions are saved to
`pretrained_model/pugan/test/4X`.

## Training

All three dataset configurations use **100 epochs** by default:

```bash
python train.py --dataset pugan --exp_name pugan
python train.py --dataset pu1k --exp_name pu1k
python train.py --dataset Sketchfab --exp_name Sketchfab
```

The default training configuration is seed 21, Adam, batch size 32, initial
learning rate `1e-3`, no weight decay, and StepLR with step size 20 and decay
factor 0.5. Training uses 256-point inputs and 1,024-point targets. The query
perturbation standard deviation is 0.02, the GFEB neighborhood size is
`k=16`, the DSA neighborhood sizes are `{2, 4, 8}`, and the feature width is
32. Checkpoints are saved every 10 epochs under `output/`.

Dataset-specific options can be passed on the same command line. For example:

```bash
python train.py --dataset pugan \
  --h5_file_path /path/to/PUGAN_poisson_256_poisson_1024.h5 \
  --epochs 100 --batch_size 32
```

## Evaluation using the HFCI-PU protocol

The `pc_eval` directory is adapted from
[HFCI-PU](https://github.com/xiaolongTang163/HFCI-PU), commit
`cb257fea1f0e28d71a92f83050f7ee8e320e0c54`. It implements the evaluation
protocol used for the manuscript results.

Install CMake and CGAL, then compile the CUDA extensions and the P2F program:

```bash
bash pc_eval/install.sh
```

Prediction and ground-truth directories must contain identically named `.xyz`
files with the same number of points. The mesh directory must contain a
matching `.off` file for every object.

PU-GAN 4x example:

```bash
python -m pc_eval.eval \
  --pred_dir ./pretrained_model/pugan/test/4X \
  --gt_dir ./data/PU-GAN/test_pointcloud/input_2048_4X/gt_8192 \
  --mesh_dir ./data/PU-GAN/test \
  --output_dir ./results/pugan_4X
```

PU1K 4x example:

```bash
python -m pc_eval.eval \
  --pred_dir ./pretrained_model/pu1k/test/4X \
  --gt_dir ./data/PU1K/test/input_2048/gt_8192 \
  --mesh_dir ./data/PU1K/test/original_meshes \
  --output_dir ./results/pu1k_4X
```

For 16x evaluation, change the prediction directory to `16X` and use the
corresponding `gt_32768` directory. Sketchfab can be evaluated with the same
command by supplying its prediction, ground-truth, and reference-mesh paths.

Following HFCI-PU, prediction and ground-truth point clouds are independently
centered and normalized to the unit sphere before CD, HD, and EMD are
computed. P2F is measured against the reference mesh. The `scaled_average`
row in `eval.csv` uses the manuscript convention: CD and HD are multiplied by
`1e3`, EMD by `1e2`, and P2F-avg/P2F-std by `1e3`. See
[`pc_eval/README.md`](pc_eval/README.md) for details.

## Reproducibility notes

- Inference uses 1,024-point patches, patch-rate 3, and 10 refinement
  iterations by default.
- The default refinement step size is 50 for PU-GAN and 500 for PU1K and
  Sketchfab.
- No smoothing or denoising is applied after upsampling. FPS is used to merge
  overlapping refined patches and retain the target number of points.
- Random seeds control NumPy, PyTorch, and Python randomness. Exact GPU
  reproducibility can still depend on CUDA and operator versions.

## Acknowledgments and license

ANCF-Net is derived in part from
[Grad-PU](https://github.com/yunhe20/Grad-PU). The evaluation code is adapted
from [HFCI-PU](https://github.com/xiaolongTang163/HFCI-PU). Both upstream
projects are distributed under the Apache License 2.0. See `LICENSE` and
`NOTICE` for attribution.

## Citation

```bibtex
@article{ancfnet2026,
  title={ANCF-Net: Adaptive Neighborhood Context Fusion with Dynamic Scale Modeling for High-Fidelity Point Cloud Upsampling},
  author={Li, Jinling and Li, Weigang and Wang, Yongqiang and Zhao, Yuntao},
  journal={The Visual Computer},
  year={2026}
}
```
