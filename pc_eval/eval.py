"""Evaluate ANCF-Net point clouds with the HFCI-PU metric protocol.

This module is adapted from HFCI-PU (Apache License 2.0). It computes CD,
HD, EMD, P2F-avg, and P2F-std after independently normalizing prediction and
ground-truth point clouds to the unit sphere.
"""

import argparse
import csv
import os
import subprocess
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pc_eval.common.loss import Loss
from pc_eval.utils.pc_util import load, normalize_point_cloud


def compute_p2f(prediction_path, mesh_path):
    executable = Path(__file__).resolve().parent / "evaluate_code" / "evaluate"
    if not executable.is_file():
        raise FileNotFoundError(
            f"P2F executable not found at {executable}. Run bash pc_eval/install.sh first."
        )
    subprocess.run(
        [str(executable), str(mesh_path), str(prediction_path)],
        stdout=subprocess.DEVNULL,
        check=True,
    )


def evaluate(pred_dir, gt_dir, mesh_dir, output_dir):
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    mesh_dir = Path(mesh_dir)
    output_dir = Path(output_dir)

    for directory, label in ((pred_dir, "prediction"), (gt_dir, "ground truth"), (mesh_dir, "mesh")):
        if not directory.is_dir():
            raise FileNotFoundError(f"The {label} directory does not exist: {directory}")

    gt_paths = sorted(gt_dir.glob("*.xyz"))
    if not gt_paths:
        raise ValueError(f"No .xyz ground-truth files were found in {gt_dir}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("The HFCI-PU CD/HD/EMD extensions require a CUDA-enabled PyTorch environment.")
    metric = Loss().cuda()

    for gt_path in tqdm(gt_paths, desc="Computing P2F"):
        pred_path = pred_dir / gt_path.name
        mesh_path = mesh_dir / f"{gt_path.stem}.off"
        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing prediction: {pred_path}")
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Missing mesh: {mesh_path}")
        p2f_path = pred_path.with_name(f"{pred_path.stem}_point2mesh_distance.txt")
        if not p2f_path.is_file():
            compute_p2f(pred_path, mesh_path)

    fieldnames = ["name", "CD", "HD", "P2F-avg", "P2F-std", "EMD"]
    raw_rows = []
    all_p2f = []

    for gt_path in tqdm(gt_paths, desc="Computing CD/HD/EMD"):
        pred_path = pred_dir / gt_path.name
        pred_xyz = load(pred_path)[:, :3]
        gt_xyz = load(gt_path)[:, :3]
        if pred_xyz.shape[0] != gt_xyz.shape[0]:
            raise ValueError(
                f"EMD requires equal point counts, but {pred_path.name} contains "
                f"{pred_xyz.shape[0]} points and its ground truth contains {gt_xyz.shape[0]}."
            )
        pred_xyz, _, _ = normalize_point_cloud(pred_xyz)
        gt_xyz, _, _ = normalize_point_cloud(gt_xyz)

        pred_tensor = torch.from_numpy(pred_xyz).unsqueeze(0).contiguous().cuda()
        gt_tensor = torch.from_numpy(gt_xyz).unsqueeze(0).contiguous().cuda()

        cd = metric.get_cd_loss(pred_tensor, gt_tensor).cpu().item()
        hd = metric.get_hd_loss(pred_tensor, gt_tensor).cpu().item()
        emd = metric.get_emd_loss(pred_tensor, gt_tensor).cpu().item()

        p2f_path = pred_path.with_name(f"{pred_path.stem}_point2mesh_distance.txt")
        p2f_values = np.atleast_2d(load(p2f_path))[:, 3]
        p2f_values = p2f_values[np.isfinite(p2f_values)]
        if p2f_values.size == 0:
            raise ValueError(f"No finite P2F values were found in {p2f_path}.")
        all_p2f.append(p2f_values)

        raw_rows.append(
            OrderedDict(
                name=gt_path.stem,
                CD=cd,
                HD=hd,
                EMD=emd,
                **{"P2F-avg": float(np.mean(p2f_values)), "P2F-std": float(np.std(p2f_values))},
            )
        )

    p2f_global = np.concatenate(all_p2f)
    average = OrderedDict(
        name="average",
        CD=float(np.mean([row["CD"] for row in raw_rows])),
        HD=float(np.mean([row["HD"] for row in raw_rows])),
        EMD=float(np.mean([row["EMD"] for row in raw_rows])),
        **{"P2F-avg": float(np.mean(p2f_global)), "P2F-std": float(np.std(p2f_global))},
    )
    scaled = OrderedDict(
        name="scaled_average",
        CD=round(average["CD"] * 1e3, 3),
        HD=round(average["HD"] * 1e3, 3),
        EMD=round(average["EMD"] * 1e2, 3),
        **{
            "P2F-avg": round(average["P2F-avg"] * 1e3, 3),
            "P2F-std": round(average["P2F-std"] * 1e3, 3),
        },
    )

    csv_path = output_dir / "eval.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in raw_rows + [average, scaled]:
            writer.writerow(row)

    result_path = output_dir / "finalresult.txt"
    result_path.write_text(f"{average}\n{scaled}\n", encoding="utf-8")
    print(scaled)
    print(f"Detailed results: {csv_path}")
    return scaled


def parse_args():
    parser = argparse.ArgumentParser(description="ANCF-Net point-cloud evaluation")
    parser.add_argument("--pred_dir", required=True, help="directory containing predicted .xyz files")
    parser.add_argument("--gt_dir", required=True, help="directory containing ground-truth .xyz files")
    parser.add_argument("--mesh_dir", required=True, help="directory containing reference .off meshes")
    parser.add_argument("--output_dir", required=True, help="directory for eval.csv and finalresult.txt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.pred_dir, args.gt_dir, args.mesh_dir, args.output_dir)
