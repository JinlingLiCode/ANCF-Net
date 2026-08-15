import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import get_knn_pts, index_points
from einops import repeat

class interpolate_feature_weight(nn.Module):
    def __init__(self, args):
        super(interpolate_feature_weight, self).__init__()

        self.mlp_0 = Conv2D(3, args.feat_dim)
        self.mlp_1 = Conv2D(args.feat_dim, args.feat_dim)
        self.mlp_2 = Conv2D(args.feat_dim*2, 1, with_bn=False, with_relu=False)
        self.k = args.k
        self.act = torch.nn.Sigmoid()


    def forward(self, original_pts, query_pts, local_feat):
        B, _, N = query_pts.shape
        knn_pts, knn_idx = get_knn_pts(self.k, original_pts, query_pts, return_idx=True)
        patch_feat = index_points(local_feat, knn_idx)[:,:,:,:self.k//4] # b, c, n, k/4   前四分之一k的点的特征
        
        repeat_query_pts = repeat(query_pts, 'b c n -> b c n k', k=self.k)
        relative_pts = knn_pts - repeat_query_pts # b 3 n k    k近邻和查询点的差值
        feat0 = self.mlp_0(relative_pts)# b c n k   根据距离进行特征提取
        #torch.max()返回两个张量：
        # 第一个张量包含每个特征通道上的最大值。
        # 第二个张量包含每个最大值在特征通道上的索引
        feat_g = torch.max(feat0, dim=3, keepdim=True)[0]# b c n 1       取前k的点的特征值最大的点？ 64 32 1024 1  相当于最大池化提取最大特征
        relative_feat = self.mlp_1(feat0)[:,:,:,:self.k//4] # b c n k/4
        feat2 = torch.cat([relative_feat, feat_g.view(B, -1, N, 1).repeat(1, 1, 1, self.k//4),], dim=1) # b 2c n k/4
        weight = self.act(self.mlp_2(feat2)) # b 1 n k/4
        weight_d = 1-weight   #  距离越近  权值越大
        
        query_feat = weight_d * relative_feat + weight * patch_feat
        query_feat = torch.sum(query_feat, dim=-1) # (b, c, n)
        return query_feat


class interpolate_feature_point_Multi_scale_adaptive_Knn_attention(nn.Module):
    def __init__(self, args):
        super(interpolate_feature_point_Multi_scale_adaptive_Knn_attention, self).__init__()

        self.k_values = [2,4,  8]  # 定义不同大小的k近邻个
        self.mlp_0 = nn.ModuleList([Conv2D(3, args.feat_dim) for _ in self.k_values])
        self.mlp_1 = nn.ModuleList([Conv2D(args.feat_dim, args.feat_dim) for _ in self.k_values])
        self.mlp_2 = nn.ModuleList(
            [Conv2D(args.feat_dim * 2, 1, with_bn=False, with_relu=False) for _ in self.k_values])
        self.mlp_3 = nn.ModuleList(
            [Conv1D(args.feat_dim, 1, with_bn=False, with_relu=False) for _ in self.k_values])
        self.act = torch.nn.Sigmoid()



    def forward(self, original_pts, query_pts, local_feat):
        B, C, N = query_pts.shape
        query_feat_list = []
        query_feat_weight_list = []

        for i, k in enumerate(self.k_values):
            knn_pts, knn_idx = get_knn_pts(k, original_pts, query_pts, return_idx=True)
            patch_feat = index_points(local_feat, knn_idx)  # b, c, n, k

            repeat_query_pts = query_pts.unsqueeze(-1).repeat(1, 1, 1, k)
            relative_pts = knn_pts - repeat_query_pts  # b 3 n k
            feat0 = self.mlp_0[i](relative_pts)  # b c n k
            feat_g = torch.max(feat0, dim=3, keepdim=True)[0]  # b c n 1
            relative_feat = self.mlp_1[i](feat0)  # b c n k

            feat2 = torch.cat([relative_feat, feat_g.view(B, -1, N, 1).repeat(1, 1, 1, k)], dim=1)  # b 2c n k
            weight = self.act(self.mlp_2[i](feat2))  # b 1 n k
            weight_d = 1 - weight

            query_feat = weight_d * relative_feat + weight * patch_feat
            query_feat = torch.sum(query_feat, dim=-1)  # (b, c, n)
            query_feat_weight = self.mlp_3[i](query_feat)   # (b, 1, n)
            query_feat_weight_list.append(query_feat_weight) # (b, 1, n)   list
            query_feat = query_feat.unsqueeze(dim=1)  # (b, 1，c, n)

            query_feat_list.append(query_feat)

        matrix = torch.cat([query_feat_weight_list[i] for i in range(len(self.k_values))], dim=1)
        matrix = F.softmax(matrix, dim=1)
        matrix = matrix.unsqueeze(dim=2)

        query_feat = torch.cat([query_feat_list[i] for i in range(len(self.k_values))], dim=1)  # (b, 3，c, n)
        # query_feat = torch.sum(query_feat, dim=1)
        query_feat = (matrix * query_feat).sum(dim=1)

        return query_feat



class Conv1D(nn.Module):
    def __init__(self, input_dim, output_dim, with_bn=True, with_relu=True):
        super(Conv1D, self).__init__()
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.conv = nn.Conv1d(input_dim, output_dim, 1)
        if with_bn:
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        """
            x: (B, C, N)
        """
        if self.with_bn:
            x = self.bn(self.conv(x))
        else:
            x = self.conv(x)

        if self.with_relu:
            x = F.relu(x)
        return x
    
class Conv2D(nn.Module):
    def __init__(self, input_dim, output_dim, with_bn=True, with_relu=True):
        super(Conv2D, self).__init__()
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.conv = nn.Conv2d(input_dim, output_dim, 1)
        if with_bn:
            self.bn = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        """
            x: (B, C, N)
        """
        if self.with_bn:
            x = self.bn(self.conv(x))
        else:
            x = self.conv(x)

        if self.with_relu:
            x = F.relu(x)
        return x
