import torch
import math
from einops import rearrange
from models.pointops.functions import pointops
import logging
import os
import numpy as np
import random
from torch.autograd import grad
from einops import rearrange, repeat
from sklearn.neighbors import NearestNeighbors
from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
# from models.pointasnl_utils import *
import torch.nn as nn
import torch.nn.functional as F
chamfer_dist = chamfer_3DDist()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def index_points(pts, idx):
    """
    Input:
        pts: input points data, [B, C, N]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, C, S, [K]]
    """
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = pts.shape[1]
    reshape = False
    if len(idx.shape) == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1)
    # (b, c, (s k))
    res = torch.gather(pts, 2, idx[:, None].repeat(1, fdim, 1))
    if reshape:
        res = rearrange(res, 'b c (s k) -> b c s k', s=sample_num)

    return res


def FPS(pts, fps_pts_num):
    # input: (b, 3, n)

    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, fps_pts_num)
    sample_idx = pointops.furthestsampling(pts_trans, fps_pts_num).long()
    # (b, 3, fps_pts_num)
    sample_pts = index_points(pts, sample_idx)

    return sample_pts


def get_knn_pts(k, pts, center_pts, return_idx=False):
    # input: (b, 3, n)

    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, m, 3)
    center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()
    # (b, m, k)
    knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()
    # (b, 3, m, k)
    knn_pts = index_points(pts, knn_idx)

    if return_idx == False:
        return knn_pts
    else:
        return knn_pts, knn_idx


def midpoint_interpolate(args, sparse_pts):
    # sparse_pts: (b, 3, n)

    pts_num = sparse_pts.shape[-1]

    up_pts_num = int(pts_num * args.up_rate)
    k = int(2 * args.up_rate)
    # (b, 3, n, k)
    knn_pts = get_knn_pts(k, sparse_pts, sparse_pts)
    # (b, 3, n, k)
    repeat_pts = repeat(sparse_pts, 'b c n -> b c n k', k=k)
    # (b, 3, n, k)
    mid_pts = (knn_pts + repeat_pts) / 2.0
    # (b, 3, (n k))
    mid_pts = rearrange(mid_pts, 'b c n k -> b c (n k)')
    # note that interpolated_pts already contain sparse_pts
    interpolated_pts = mid_pts
    # fps: (b, 3, up_pts_num)
    interpolated_pts = FPS(interpolated_pts, up_pts_num)

    return interpolated_pts


def midpoint_interpolate_v1(args, sparse_pts):  #生成更多点云  不仅仅是生成下采样目标数量点云
    # sparse_pts: (b, 3, n)

    pts_num = sparse_pts.shape[-1]
    up_rate = 2 * args.up_rate
    up_pts_num = int(pts_num * up_rate)
    k = int(2 * up_rate)
    # (b, 3, n, k)
    knn_pts = get_knn_pts(k, sparse_pts, sparse_pts)
    # (b, 3, n, k)
    repeat_pts = repeat(sparse_pts, 'b c n -> b c n k', k=k)
    # (b, 3, n, k)
    mid_pts = (knn_pts + repeat_pts) / 2.0
    # (b, 3, (n k))
    mid_pts = rearrange(mid_pts, 'b c n k -> b c (n k)')
    # note that interpolated_pts already contain sparse_pts
    interpolated_pts = mid_pts
    # fps: (b, 3, up_pts_num)
    interpolated_pts = FPS(interpolated_pts, up_pts_num)

    return interpolated_pts

def generate_interpolated_point_cloud(original_points, query_points, k=3):
    """
    根据查询点在原始点附近的 k 个点，生成新的插值点。

    Parameters:
    original_points (ndarray): 原始点云数据，形状为 (n_points, 3)
    query_points (ndarray): 查询点，形状为 (m_points, 3)
    k (int): 用于插值的邻近点的数量

    Returns:
    interpolated_points (ndarray): 插值生成的点，形状为 (m_points, 3)
    """

    # 构建KDTree以查找邻近点
    tree = KDTree(original_points)

    # 查找每个查询点的 k 个最近邻点
    distances, indices = tree.query(query_points, k=k)

    # 插值结果初始化
    interpolated_points = np.zeros_like(query_points)

    # 使用线性插值方法进行插值
    for i, idx in enumerate(indices):
        neighbors = original_points[idx]
        weights = 1 / (distances[i] + 1e-6)  # 防止除零错误
        weights /= weights.sum()
        interpolated_points[i] = np.dot(weights, neighbors)

    return interpolated_points



def midpoint_interpolate_noFPS(args, sparse_pts):
    # sparse_pts: (b, 3, n)

    pts_num = sparse_pts.shape[-1]
    up_pts_num = int(pts_num * args.up_rate)
    k = int(2 * args.up_rate)
    # (b, 3, n, k)
    knn_pts = get_knn_pts(k, sparse_pts, sparse_pts)
    # (b, 3, n, k)
    repeat_pts = repeat(sparse_pts, 'b c n -> b c n k', k=k)
    # (b, 3, n, k)
    mid_pts = (knn_pts + repeat_pts) / 2.0
    # (b, 3, (n k))
    mid_pts = rearrange(mid_pts, 'b c n k -> b c (n k)')
    # note that interpolated_pts already contain sparse_pts
    interpolated_pts = mid_pts
    # fps: (b, 3, up_pts_num)
    # interpolated_pts = FPS(interpolated_pts, up_pts_num)

    return interpolated_pts


def k_nearest_neighbor_interpolation(args, sparse_pts):
    # sparse_pts: (b, 3, n)

    pts_num = sparse_pts.shape[-1]
    up_pts_num = int(pts_num * args.up_rate)
    k = int(2 * args.up_rate)  # K nearest neighbors
    # (b, 3, n, k)
    knn_pts, _ = get_knn_pts(k, sparse_pts, sparse_pts, return_idx=True)
    # (b, 3, n, 1)
    sparse_pts = sparse_pts.unsqueeze(-1)
    # (b, 3, n, k)
    sparse_pts = sparse_pts.repeat(1, 1, 1, k)
    # (b, 3, n, k)
    interpolated_pts = sparse_pts + knn_pts
    # (b, 3, (n k))
    interpolated_pts = rearrange(interpolated_pts, 'b c n k -> b c (n k)')
    # note that interpolated_pts already contain sparse_pts
    # randomly sample up_pts_num points using FPS
    interpolated_pts = FPS(interpolated_pts, up_pts_num)

    return interpolated_pts


def get_p2p_loss(args, pred_p2p, sample_pts, gt_pts):
    # input: (b, c, n)

    # (b, 3, n)
    knn_pts = get_knn_pts(1, gt_pts, sample_pts).squeeze(-1)
    # (b, 1, n)
    gt_p2p = torch.norm(knn_pts - sample_pts, p=2, dim=1, keepdim=True)
    # (b, 1, n)

    if args.use_smooth_loss == True:
        if args.truncate_distance == True:
            loss = torch.nn.SmoothL1Loss(reduction='none', beta=args.beta)(torch.clamp(pred_p2p, max=args.max_dist), torch.clamp(gt_p2p, max=args.max_dist))
        else:
            loss = torch.nn.SmoothL1Loss(reduction='none', beta=args.beta)(pred_p2p, gt_p2p)
    else:
        if args.truncate_distance == True:
            loss = torch.nn.L1Loss(reduction='none')(torch.clamp(pred_p2p, max=args.max_dist), torch.clamp(gt_p2p, max=args.max_dist))
        else:
            loss = torch.nn.L1Loss(reduction='none')(pred_p2p, gt_p2p)
    # (b, 1, n) -> (b, n) -> (b) -> scalar
    loss = loss.squeeze(1).sum(dim=-1).mean()

    return loss

def get_p2p_loss_with_direction(args, pred_p2p, sample_pts, gt_pts):
    # input: (b, c, n)

    # (b, 3, n)
    knn_pts = get_knn_pts(1, gt_pts, sample_pts).squeeze(-1)
    # (b, 1, n)
    gt_p2p = torch.norm(knn_pts - sample_pts, p=2, dim=1, keepdim=True)
    # (b, 1, n)

    # 将符号方向赋给pred_p2p
    direction_vectors = (knn_pts - sample_pts) / gt_p2p  # 计算方向向量


    if args.use_smooth_loss == True:
        if args.truncate_distance == True:
            loss = torch.nn.SmoothL1Loss(reduction='none', beta=args.beta)(torch.clamp(pred_p2p, max=args.max_dist), torch.clamp(gt_p2p, max=args.max_dist))
        else:
            loss = torch.nn.SmoothL1Loss(reduction='none', beta=args.beta)(pred_p2p, gt_p2p)
    else:
        if args.truncate_distance == True:
            loss = torch.nn.L1Loss(reduction='none')(torch.clamp(pred_p2p, max=args.max_dist), torch.clamp(gt_p2p, max=args.max_dist))
        else:
            # 将真实距离也转换为向量形式
            signed_gt_p2p = direction_vectors * gt_p2p
            loss = torch.nn.L1Loss(reduction='none')(pred_p2p, signed_gt_p2p)


    # (b, 1, n) -> (b, n) -> (b) -> scalar
    loss = loss.squeeze(1).sum(dim=-1).mean()


    return loss

def chamfer_distance(pc1, pc2):
    """
    计算两个点云之间的倒角距离
    """
    dist1 = torch.cdist(pc1, pc2)
    dist2 = torch.cdist(pc2, pc1)

    min_dist1 = torch.min(dist1, dim=2)[0]
    min_dist2 = torch.min(dist2, dim=2)[0]

    loss = torch.mean(min_dist1) + torch.mean(min_dist2)
    return loss

def uniform_loss(pc, radius):
    """
    计算点云的均匀损失
    """
    knn_distances = torch.cdist(pc, pc)
    mask = knn_distances < radius
    uniform_loss = torch.mean(mask.float()) - torch.mean((knn_distances[mask]).float())
    return uniform_loss

def get_p2p_loss_fusion(args, pred_p2p, sample_pts, gt_pts):
    # input: (b, c, n)

    # 获取最近邻点
    knn_pts = get_knn_pts(1, gt_pts, sample_pts).squeeze(-1)

    # 计算点到点的欧氏距离
    gt_p2p = torch.norm(knn_pts - sample_pts, p=2, dim=1, keepdim=True)

    # 选择损失函数
    if args.use_smooth_loss:
        if args.truncate_distance:
            loss = torch.nn.SmoothL1Loss(reduction='none', beta=args.beta)(
                torch.clamp(pred_p2p, max=args.max_dist),
                torch.clamp(gt_p2p, max=args.max_dist)
            )
        else:
            loss = torch.nn.SmoothL1Loss(reduction='none', beta=args.beta)(pred_p2p, gt_p2p)
    else:
        if args.truncate_distance:
            loss = torch.nn.L1Loss(reduction='none')(
                torch.clamp(pred_p2p, max=args.max_dist),
                torch.clamp(gt_p2p, max=args.max_dist)
            )
        else:
            loss = torch.nn.L1Loss(reduction='none')(pred_p2p, gt_p2p)

    # (b, 1, n) -> (b, n) -> (b) -> scalar
    loss = loss.squeeze(1).sum(dim=-1).mean()

    # 使用pred_p2p生成预测点云
    direction_vectors = (knn_pts - sample_pts) / gt_p2p  # 计算方向向量
    pred_pts = sample_pts + direction_vectors * pred_p2p  # 生成新的预测点云

    args.lambda_cd = 0
    args.lambda_uniform = 10
    args.uniform_radius = 0.1

    # 计算倒角距离
    cd_loss = chamfer_distance(pred_pts, gt_pts)

    # 计算均匀损失
    radius = args.uniform_radius if hasattr(args, 'uniform_radius') else 0.1  # 使用给定的半径，或者默认值0.1
    uniform_loss_value = uniform_loss(pred_pts, radius)

    # 总损失
    total_loss = loss + args.lambda_cd * cd_loss + args.lambda_uniform * uniform_loss_value
    # total_loss = loss + args.lambda_cd * cd_loss
    return total_loss, loss, cd_loss, uniform_loss_value



def normalize_point_cloud(input, centroid=None, furthest_distance=None):
    # input: (b, 3, n) tensor

    if centroid is None:
        # (b, 3, 1)
        centroid = torch.mean(input, dim=-1, keepdim=True)
    # (b, 3, n)
    input = input - centroid
    if furthest_distance is None:
        # (b, 3, n) -> (b, 1, n) -> (b, 1, 1)
        furthest_distance = torch.max(torch.norm(input, p=2, dim=1, keepdim=True), dim=-1, keepdim=True)[0]
    input = input / furthest_distance

    return input, centroid, furthest_distance


def add_noise(pts, sigma, clamp):
    # input: (b, 3, n)

    assert (clamp > 0)
    jittered_data = torch.clamp(sigma * torch.randn_like(pts), -1 * clamp, clamp).cuda()
    jittered_data += pts

    return jittered_data


# generate patch for test   由center_pts对pts进行k近邻采样，生成center_pts个patch
def extract_knn_patch(k, pts, center_pts):
    # input : (b, 3, n)     center_pts是由fps提取的有代表性的点

    # (n, 3)
    pts_trans = rearrange(pts.squeeze(0), 'c n -> n c').contiguous()
    pts_np = pts_trans.detach().cpu().numpy()
    # (m, 3)
    center_pts_trans = rearrange(center_pts.squeeze(0), 'c m -> m c').contiguous()
    center_pts_np = center_pts_trans.detach().cpu().numpy()
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pts_np)
    # (m, k)
    knn_idx = knn_search.kneighbors(center_pts_np, return_distance=False)
    # (m, k, 3)
    patches = np.take(pts_np, knn_idx, axis=0)
    patches = torch.from_numpy(patches).float().cuda()
    # (m, 3, k)
    patches = rearrange(patches, 'm k c -> m c k').contiguous()

    return patches


def get_logger(name, log_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')
    # output to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # output to log file
    log_name = name + '_log.txt'
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_query_points(input_pts, args):
    query_pts = input_pts + (torch.randn_like(input_pts) * args.local_sigma)

    return query_pts


def reset_model_args(train_args, model_args):
    for arg in vars(train_args):
        setattr(model_args, arg, getattr(train_args, arg))