
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .visualize import vis_points_3d


def knn(x: torch.tensor, k: int) -> int:
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    else:
        idx_base = (
            torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1)
            * num_points
        )
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx


def local_cov(pts: torch.tensor, idx: int) -> torch.tensor:
    batch_size = pts.size(0)
    num_points = pts.size(2)
    pts = pts.view(batch_size, -1, num_points)  # (batch_size, 3, num_points)

    _, num_dims, _ = pts.size()

    x = pts.transpose(2, 1).contiguous()  # (batch_size, num_points, 3)
    x = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*num_points*2, 3)
    x = x.view(batch_size, num_points, -1, num_dims)  # (batch_size, num_points, k, 3)

    x = torch.matmul(x[:, :, 0].unsqueeze(3), x[:, :, 1].unsqueeze(2))
    # (batch_size, num_points, 3, 1) * (batch_size, num_points, 1, 3)
    # -> (batch_size, num_points, 3, 3)

    # x = torch.matmul(x[:,:,1:].transpose(3, 2), x[:,:,1:])
    x = x.view(batch_size, num_points, 9).transpose(2, 1)  # (batch_size, 9, num_points)

    x = torch.cat((pts, x), dim=1)  # (batch_size, 12, num_points)

    return x


def local_maxpool(x: torch.tensor, idx: int) -> torch.tensor:
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims)
    x = x.transpose(2, 1).contiguous()

    # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    x = x.view(batch_size * num_points, -1)[idx, :]

    # (batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, -1, num_dims)

    # (batch_size, num_points, num_dims)
    x, _ = torch.max(x, dim=2)
    return x


def get_graph_feature(x: torch.tensor, k=20, idx=None) -> torch.tensor:
    batch_size = x.size(0)
    num_points = x.size(2)

    # (batch_size, num_dims, num_points)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        # (batch_size, num_points, k)
        idx = knn(x, k=k)

    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims)
    x = x.transpose(2, 1).contiguous()

    # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]

    # (batch_size, num_points, k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)

    # (batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    # (batch_size, 2*num_dims, num_points, k)
    return feature


class FoldingNetEncoder(nn.Module):
    def __init__(self, n_points: int, feat_dims: int) -> None:
        super().__init__()
        self.n_points = n_points
        self.k = 16
        self.mlp1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, feat_dims, 1),
        )

    def graph_layer(self, x: torch.tensor, idx: int) -> torch.tensor:
        x = local_maxpool(x, idx)
        x = self.linear1(x)
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = local_maxpool(x, idx)
        x = self.linear2(x)
        x = x.transpose(2, 1)
        x = self.conv2(x)
        return x

    def forward(self, pts: torch.tensor) -> torch.tensor:
        # (batch_size, 3, num_points)
        pts = pts.transpose(2, 1)
        idx = knn(pts, k=self.k)

        # (batch_size, 3, num_points) -> (batch_size, 12, num_points])
        x = local_cov(pts, idx)

        # (batch_size, 12, num_points) -> (batch_size, 64, num_points])
        x = self.mlp1(x)

        # (batch_size, 64, num_points) -> (batch_size, 1024, num_points)
        x = self.graph_layer(x, idx)

        # (batch_size, 1024, num_points) -> (batch_size, 1024, 1)
        x = torch.max(x, 2, keepdim=True)[0]

        # (batch_size, 1024, 1) -> (batch_size, feat_dims, 1)
        x = self.mlp2(x)

        # (batch_size, feat_dims, 1) -> (batch_size, 1, feat_dims)
        feat = x.transpose(2, 1)

        return feat  # (batch_size, 1, feat_dims)


class FoldingNetDecoder(nn.Module):
    def __init__(self, feat_dims: int, shape="plane") -> None:
        super().__init__()
        # self.m = 2048  # 45 * 45.
        self.m = 1024  # 45 * 45.
        self.shape = shape
        self.sphere = np.load(f"/opt/data/private/selfrecon/point_cloud_anomaly_detection-main/grids/sphere_{self.m}.npy")
        self.gaussian = np.load("/opt/data/private/selfrecon/point_cloud_anomaly_detection-main/grids/gaussian.npy")
        self.meshgrid = [[-0.3, 0.3, 32], [-0.6, 0.6, 64]]
        if self.shape == "plane":
            self.folding1 = nn.Sequential(
                nn.Conv1d(feat_dims + 2, feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(feat_dims, feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(feat_dims, 3, 1),
            )
        else:
            self.folding1 = nn.Sequential(
                nn.Conv1d(feat_dims + 3, feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(feat_dims, feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(feat_dims, 3, 1),
            )
        self.folding2 = nn.Sequential(
            nn.Conv1d(feat_dims + 3, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, 3, 1),
        )

    def build_grid(self, batch_size: int) -> torch.tensor:
        # assert self.shape == "plane", "shape should be 'plane'."
        if self.shape == "plane":
            x = np.linspace(*self.meshgrid[0])
            y = np.linspace(*self.meshgrid[1])
            points = np.array(list(itertools.product(x, y)))
        elif self.shape == "sphere":
            points = self.sphere
        elif self.shape == "gaussian":
            points = self.gaussian

        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)

        return points.float()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x.transpose(1, 2).repeat(1, 1, self.m)
        points = self.build_grid(x.shape[0]).transpose(1, 2)
        # vis_points_3d(points[0].transpose(0, 1), "meshgrid.png")

        if x.get_device() != -1:
            points = points.cuda(x.get_device())

        cat1 = torch.cat((x, points), dim=1)
        folding_result1 = self.folding1(cat1)
        # vis_points_3d(folding_result1[0].transpose(0, 1), "folding1.png")

        cat2 = torch.cat((x, folding_result1), dim=1)
        folding_result2 = self.folding2(cat2)
        # print(folding_result2.shape)
        # vis_points_3d(folding_result2[0].transpose(0, 1), "folding2.png")

        return folding_result2.transpose(1, 2), folding_result1.transpose(1, 2)


class FoldingNet(nn.Module):
    def __init__(self, n_points: int, feat_dims: int, shape: str) -> None:
        super().__init__()
        self.encoder = FoldingNetEncoder(n_points, feat_dims)
        self.decoder = FoldingNetDecoder(feat_dims, shape=shape)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input: torch.tensor):
        feature = self.encoder(input)
        # feature = self.softmax(feature)
        folding2, folding1 = self.decoder(feature)
        return folding2, folding1, feature

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())


class SkipFoldingNetEncoder(nn.Module):
    def __init__(self, n_points: int, feat_dims: int) -> None:
        super().__init__()
        self.n_points = n_points
        self.k = 16
        self.mlp1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024 + 64, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, feat_dims, 1),
        )

    def graph_layer(self, x: torch.tensor, idx: int) -> torch.tensor:
        x = local_maxpool(x, idx)
        x = self.linear1(x)
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = local_maxpool(x, idx)
        x = self.linear2(x)
        x = x.transpose(2, 1)
        x = self.conv2(x)
        return x

    def forward(self, pts: torch.tensor) -> torch.tensor:
        pts = pts.transpose(2, 1)
        idx = knn(pts, k=self.k)
        x = local_cov(pts, idx)
        x = self.mlp1(x)
        local_feat_1 = x
        x = self.graph_layer(x, idx)
        local_feat_2 = x
        cat_feat = torch.cat([local_feat_1, local_feat_2], 1)
        x = torch.max(cat_feat, 2, keepdim=True)[0]
        x = self.mlp2(x)
        feat = x.transpose(2, 1)

        return feat


class SkipFoldingNet(nn.Module):
    def __init__(self, n_points: int, feat_dims: int, shape: str) -> None:
        super().__init__()
        self.encoder = SkipFoldingNetEncoder(n_points, feat_dims)
        self.decoder = FoldingNetDecoder(feat_dims, shape=shape)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input: torch.tensor):
        feature = self.encoder(input)
        # feature = self.softmax(feature)
        folding2, folding1 = self.decoder(feature)
        return folding2, folding1, feature

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())


class SkipVariationalEncoder(nn.Module):
    def __init__(self, n_points: int, feat_dims: int) -> None:
        super().__init__()
        self.n_points = n_points
        self.k = 16
        self.feat_dims = feat_dims
        self.mlp1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, 1024, 1)

        # self.fc_mu = nn.Sequential(
        #     nn.Conv1d(1024 + 64, feat_dims, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(feat_dims, feat_dims, 1),
        # )
        # self.fc_var = nn.Sequential(
        #     nn.Conv1d(1024 + 64, feat_dims, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(feat_dims, feat_dims, 1),
        # )
        self.fc_mu = nn.Conv1d(1024 + 64, feat_dims, 1)
        self.fc_var = nn.Conv1d(1024 + 64, feat_dims, 1)

    def graph_layer(self, x: torch.tensor, idx: int) -> torch.tensor:
        x = local_maxpool(x, idx)
        x = self.linear1(x)
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = local_maxpool(x, idx)
        x = self.linear2(x)
        x = x.transpose(2, 1)
        x = self.conv2(x)
        return x

    def forward(self, pts: torch.tensor) -> torch.tensor:
        pts = pts.transpose(2, 1)
        idx = knn(pts, k=self.k)
        x = local_cov(pts, idx)
        x = self.mlp1(x)
        local_feat_1 = x
        x = self.graph_layer(x, idx)
        local_feat_2 = x
        cat_feat = torch.cat([local_feat_1, local_feat_2], 1)
        x = torch.max(cat_feat, 2, keepdim=True)[0]

        mu = self.fc_mu(x)
        sigma = self.fc_var(x)
        return mu, sigma


class SkipValiationalFoldingNet(nn.Module):
    def __init__(self, n_points: int, feat_dims: int, shape: str) -> None:
        super().__init__()
        self.encoder = SkipVariationalEncoder(n_points, feat_dims)
        self.decoder = FoldingNetDecoder(feat_dims, shape=shape)
        self.softmax = nn.Softmax(dim=2)

    def sample_z(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        :mu: (Tensor) Mean of the latent Gaussian
        :sigma: (Tensor) Standard deviation of the latent Gaussian
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.tensor):
        # print("input===>>",input.shape)
        mu, sigma = self.encoder(input)
        mu = mu.transpose(2, 1)
        sigma = sigma.transpose(2, 1)
        # feature = self.softmax(feature)
        feature = self.sample_z(mu, sigma)
        folding2, folding1 = self.decoder(feature)
        return folding2, folding1, mu, sigma

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.uniform_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == '__main__':

    shape = 'sphere'
    feat_dims = 512
    n_points = 1024

    model = SkipValiationalFoldingNet(n_points, feat_dims, shape)
    x = torch.randn((1,1024,3))
    output, folding1, mu, sigma = model(x)
    print("output==",output.shape," folding1==",folding1.shape)


