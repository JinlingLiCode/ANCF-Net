# ANCF-Net: Adaptive Context Fusion and Dynamic Scale Modeling for High-Fidelity Point Cloud Upsampling

 *(Note: Replace with your actual Zenodo DOI link)*

This is the official PyTorch implementation of our paper **"Adaptive Context Fusion and Dynamic Scale Modeling for High-Fidelity Point Cloud Upsampling"** (Submitted to *The Visual Computer*).

## Abstract

Point cloud upsampling is a critical preprocessing step for 3D vision tasks such as autonomous driving and surface reconstruction. Existing refinement-based methods often rely on fixed-scale neighborhood feature extraction, leading to insufficient capture of global context and local geometric details, which degrades the quality of upsampled point clouds. To address these limitations, this paper presents an adaptive neighborhood context fusion network (ANCF-Net) for high-fidelity point cloud upsampling. The proposed method employs a neighborhood context aggregator to fuse global and local features within point neighborhoods, and a dynamic scale-aware module to assign adaptive weights to multi-scale neighborhood features. These components enable the network to flexibly capture geometric structures of varying complexity. Extensive experiments on synthetic and real-world datasets show that ANCF-Net achieves state-of-the-art performance, reducing Chamfer distance and Hausdorff distance significantly compared with existing methods, while improving detail preservation and structural consistency. This work provides an effective solution for high-quality point cloud upsampling and can benefit various downstream 3D vision applications.

## Requirements

The code has been tested on Ubuntu with the following environment:

- Python >= 3.7
- PyTorch >= 1.7.1
- CUDA >= 11.0
- `numpy`, `open3d`, `einops`, `scikit-learn`, `tqdm`, `h5py`

To set up the environment, run:

Bash

```
# Create a new conda environment (optional but recommended)
conda create -n ancfnet python=3.7
conda activate ancfnet

# Install dependencies
pip install -r requirements.txt
```

### Install Custom PointOps & Chamfer Distance

You need to compile the custom CUDA operations for PointNet++ and Chamfer Distance calculation:

Bash

```
# Compile Chamfer3D
cd models/Chamfer3D
python setup.py install
cd ../..

# Compile pointops
cd models/pointops
python setup.py install
cd ../..
```

*(Optional) Evaluation Code Compilation*: If you intend to calculate standard metrics (CD, HD, P2F), please ensure the CGAL library is installed and compile the evaluation code located in the `evaluation_code` folder, following standard PU-GAN/PU-GCN evaluation protocols.

## Data Preparation

We train and evaluate our network on the widely used **PU-GAN** and **PU1K** datasets.

Please download the datasets (train sets and test meshes) and organize them into the `data/` directory. Since the PU-GAN dataset provides mesh files for testing, we generate test point clouds via Poisson disk sampling.

### Generate PU-GAN Test Point Clouds

Bash

```
# Generate 4X upsampling test data
python prepare_pugan.py --input_pts_num 2048 --gt_pts_num 8192

# Generate 16X upsampling test data
python prepare_pugan.py --input_pts_num 2048 --gt_pts_num 32768
```

*Note: You can add the `--noise_level` argument (e.g., `--noise_level 0.01`) to generate noisy inputs for robustness evaluation.*

### Directory Structure

Ensure your `data` directory looks like this:

```
data  
├───PU-GAN
│   ├───test             # test mesh files
│   ├───test_pointcloud  # generated test point clouds
│   │   ├───input_2048_4X
│   │   ├───input_2048_16X
│   │   ...
│   └───train
│       └───PUGAN_poisson_256_poisson_1024.h5
└───PU1K
    ├───test
    └───train
        └───pu1k_poisson_256_poisson_1024_pc_2500_patch50.h5 
```

## Quick Start (Evaluation)

We provide pre-trained models in the `pretrained_model` directory. You can use them directly to reproduce the results reported in the paper.

### PU-GAN Testing

Bash

```
# 4X Upsampling
python test.py --dataset pugan --test_input_path ./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048/ --ckpt_path ./pretrained_model/pugan/ckpt/best_model.pth --save_dir 4X --up_rate 4

# 16X Upsampling
python test.py --dataset pugan --test_input_path ./data/PU-GAN/test_pointcloud/input_2048_16X/input_2048/ --ckpt_path ./pretrained_model/pugan/ckpt/best_model.pth --save_dir 16X --up_rate 16
```

The upsampled point clouds will be saved in `./pretrained_model/pugan/test/save_dir`.

### PU1K Testing

Bash

```
# 4X Upsampling
python test.py --dataset pu1k --test_input_path ./data/PU1K/test/input_2048/input_2048/ --ckpt_path ./pretrained_model/pu1k/ckpt/best_model.pth --save_dir 4X --up_rate 4
```

## Training from Scratch

To train ANCF-Net from scratch on the datasets, use the following commands:

**Train on PU-GAN:**

Bash

```
python train.py --dataset pugan
```

**Train on PU1K:**

Bash

```
python train.py --dataset pu1k
```

Training logs and model checkpoints will be saved in the `./output/` directory.

## Acknowledgments

Our code implementation is inspired by and built upon several excellent open-source projects, including PU-GCN, PU-GAN, and Grad-PU. We sincerely thank the authors for making their code available to the community.

## Citation

If you find this code or our paper useful in your research, please consider citing our work:

代码段

```
@article{ancfnet2026,
  title={Adaptive Context Fusion and Dynamic Scale Modeling for High-Fidelity Point Cloud Upsampling},
  author={Li, Jinling and Li, Weigang and Wang, Yongqiang and Zhao, Yuntao},
  journal={Submitted to The Visual Computer},
  year={2026}
}
```