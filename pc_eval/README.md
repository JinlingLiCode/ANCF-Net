# Evaluation protocol

The code in this directory is adapted from the `pc_eval` implementation in
[HFCI-PU](https://github.com/xiaolongTang163/HFCI-PU), commit
`cb257fea1f0e28d71a92f83050f7ee8e320e0c54`, under the Apache License 2.0.

It computes Chamfer distance (CD), Hausdorff distance (HD), Earth Mover's
Distance (EMD), point-to-surface mean distance (P2F-avg), and point-to-surface
standard deviation (P2F-std). Prediction and ground-truth point clouds are
independently centered and normalized to the unit sphere before CD, HD, and
EMD are evaluated, following the HFCI-PU protocol.

## Build

Install CGAL and CMake, then compile the CUDA and P2F extensions from the
repository root:

```bash
bash pc_eval/install.sh
```

## Run

Prediction and ground-truth directories must contain matching `.xyz`
filenames and equal point counts (required by EMD). The mesh directory must
contain a matching `.off` file for every point cloud.

```bash
python -m pc_eval.eval \
  --pred_dir ./pretrained_model/pugan/test/4X \
  --gt_dir ./data/PU-GAN/test_pointcloud/input_2048_4X/gt_8192 \
  --mesh_dir ./data/PU-GAN/test \
  --output_dir ./results/pugan_4X
```

The script writes per-object and average metrics to `eval.csv`. The
`scaled_average` row uses the manuscript convention: CD and HD are multiplied
by `1e3`, EMD by `1e2`, and P2F-avg/P2F-std by `1e3`.
