import torch
import torch.nn as nn
from models.utils import get_knn_pts, index_points
from einops import repeat, rearrange
from models.pointops.functions import pointops
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class Point3DConv(nn.Module):
    def __init__(self, args):
        super(Point3DConv, self).__init__()

        self.k = args.k
        self.args = args
        self.conv_delta = nn.Sequential(
            nn.Conv2d(3, args.growth_rate, 1),
            nn.BatchNorm2d(args.growth_rate),
            nn.ReLU(inplace=True)
        )
        self.conv_feats = nn.Sequential(
            nn.Conv2d(args.bn_size * args.growth_rate, args.growth_rate, 1),
            nn.BatchNorm2d(args.growth_rate),
            nn.ReLU(inplace=True)
        )
        self.post_conv = nn.Sequential(
            nn.Conv2d(args.growth_rate, args.growth_rate, 1),
            nn.BatchNorm2d(args.growth_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats, pts, knn_idx=None):
        # input: (b, c, n)
        if knn_idx == None:
            # (b, 3, n, k), (b, n, k)
            knn_pts, knn_idx = get_knn_pts(self.k, pts, pts, return_idx=True)
        else:
            knn_pts = index_points(pts, knn_idx)
        # (b, 3, n, k)
        knn_delta = knn_pts - pts[..., None]           # 以采样点为中心点，取临近点相对于中心点的坐标
          ##将中心点与k近邻点差值 cat
        # (b, c, n, k)
        knn_delta = self.conv_delta(knn_delta)
        # (b, c, n, k)
        knn_feats = index_points(feats, knn_idx)
        # (b, c, n, k)
        knn_feats = self.conv_feats(knn_feats)
        # multiply: (b, c, n, k)
        new_feats = knn_delta * knn_feats
        # (b, c, n, k)
        new_feats = self.post_conv(new_feats)
        # sum: (b, c, n)  
        new_feats = new_feats.sum(dim=-1)
        # new_feats = new_feats.max(dim=-1, keepdim=False)[0]

        return new_feats

class PointTransformerConv(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super(PointTransformerConv, self).__init__()

        assert feature_dim % num_heads == 0, f"feature_dim={feature_dim} 必须能被 num_heads={num_heads} 整除！"

        self.feature_dim = feature_dim
        self.self_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

        self.norm1 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, x):
        # print(f"Input x.shape: {x.shape}")  # Debug，(B, C, N)

        x = x.permute(2, 0, 1)  # **(B, C, N) → (N, B, C)**，符合 `MultiheadAttention` 输入
        # print(f"Permuted x.shape: {x.shape}")  # Debug，(N, B, C)

        # **确保 `embed_dim` 匹配**
        # assert x.shape[-1] == self.feature_dim, f"Expected {self.feature_dim}, got {x.shape[-1]}"

        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))

        x = x.permute(1, 2, 0)  # **(N, B, C) → (B, C, N)** 还原形状
        # print(f"Output x.shape: {x.shape}")  # Debug，(B, C, N)
        return x


class DenseLayer(nn.Module):
    def __init__(self, args, input_dim):
        super(DenseLayer, self).__init__()

        self.conv_bottle = nn.Sequential(
            nn.Conv1d(input_dim, args.bn_size * args.growth_rate, 1),
            nn.BatchNorm1d(args.bn_size * args.growth_rate),
            nn.ReLU(inplace=True)
        )
        self.point_conv = Point3DConv(args)

    def forward(self, feats, pts, knn_idx=None):
        # input: (b, c, n)

        new_feats = self.conv_bottle(feats)
        # (b, c, n)
        new_feats = self.point_conv(new_feats, pts, knn_idx)
        # concat
        return torch.cat((feats, new_feats), dim=1)


class DenseUnit(nn.Module):
    def __init__(self, args):
        super(DenseUnit, self).__init__()

        self.dense_layers = nn.ModuleList([])
        for i in range(args.layer_num):
            self.dense_layers.append(DenseLayer(args, args.feat_dim + i * args.growth_rate))

    def forward(self, feats, pts, knn_idx=None):
        # input: (b, c, n)

        for dense_layer in self.dense_layers:
            new_feats = dense_layer(feats, pts, knn_idx)
            feats = new_feats
        return feats


class Transition(nn.Module):
    def __init__(self, args):
        super(Transition, self).__init__()

        input_dim = args.feat_dim + args.layer_num * args.growth_rate
        # input_dim = args.feat_dim
        self.trans = nn.Sequential(
            nn.Conv1d(input_dim, args.feat_dim, 1),
            nn.BatchNorm1d(args.feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats):
        # input: (b, c, n)

        new_feats = self.trans(feats)
        return new_feats


class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()

        self.k = args.k
        self.conv_init = nn.Sequential(
            nn.Conv1d(3, args.feat_dim, 1),
            nn.BatchNorm1d(args.feat_dim),
            nn.ReLU(inplace=True)
        )
        self.dense_blocks = nn.ModuleList([])
        for i in range(args.block_num):
            self.dense_blocks.append(nn.ModuleList([
                DenseUnit(args),
                Transition(args)
            ]))

        # # **✅ 用 `PointTransformerConv` 提取全局特征**
        # self.point_transformer = PointTransformerConv(args.feat_dim)

    def forward(self, pts):
        # input: (b, 3, n)

        # get knn_idx: (b, n, 3)
        pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
        # (b, m, k)
        knn_idx = pointops.knnquery_heap(self.k, pts_trans, pts_trans).long()
        # (b, c, n)
        init_feats = self.conv_init(pts)
        local_feats = []
        local_feats.append(init_feats)
        # local features
        for dense_block, trans in self.dense_blocks:
            new_feats = dense_block(init_feats, pts, knn_idx)
            new_feats = trans(new_feats)
            init_feats = new_feats
            local_feats.append(init_feats)
        # global features: (b, c)
        global_feats = init_feats.max(dim=-1)[0]

        # # **✅ Transformer 处理全局特征**
        # global_feats = self.point_transformer(init_feats)  # Transformer 需要 `[B, N, C]`
        # global_feats = global_feats.max(dim=-1)[0]  # (B, C)

        return global_feats, local_feats
