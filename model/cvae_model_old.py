import os

import torch
import torch.nn as nn
import pdb

class DistributionDenoiseAttention(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(DistributionDenoiseAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(in_dim, num_heads)

    def forward(self, x, distribution):
        # 计算注意力权重
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        
        # 根据分布信息调整权重
        adjusted_weights = attn_weights * distribution  # 或其他方式结合
        attn_output = torch.matmul(adjusted_weights, x)
        
        return attn_output

class MultiplicativeDenoiseAttention(nn.Module):
    def __init__(self, in_dim, c_dim):
        super(MultiplicativeDenoiseAttention, self).__init__()
        self.fc_weight = nn.Linear(c_dim, in_dim)
        self.fc_value = nn.Linear(in_dim, in_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, condition):
        # z 是潜在向量，condition 是嵌入条件
        weight = self.fc_weight(condition)  # [B, in_dim]
        value = self.fc_value(z)            # [B, in_dim]
        # print("condition==>>",condition.shape, value.shape, weight.shape)
        # 计算注意力权重并进行加权去噪
        attention_weights = self.sigmoid(weight)  # 控制噪声的权重
        denoised_output = attention_weights * value  # 加权输出
        
        return denoised_output

class GaussianAttention(nn.Module):
    def __init__(self, in_dim):
        super(GaussianAttention, self).__init__()
        self.linear = nn.Linear(in_dim, in_dim)  # 可调整维度
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        
        # 计算均值和方差
        mean = x.mean(dim=1, keepdim=True)  # [B, 1, C]
        var = x.var(dim=1, keepdim=True) + 1e-6  # 加小常数以防止除零

        # 计算高斯权重，避免指数计算溢出
        weights = torch.exp(-((x - mean) ** 2) / (2 * var + 1e-6))
        
        # 添加 NaN 检查
        if torch.isnan(weights).any():
            print("NaN in weights after exp!")

        weights = self.softmax(weights)  # 应用 softmax 以获得归一化权重
        
        # 加权特征
        out = (weights * x).sum(dim=1)  # [B, C]
        
        return out


class CombinedAttention(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(CombinedAttention, self).__init__()
        self.gaussian_attention = GaussianAttention(in_dim)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads)

    def forward(self, x):
        # 先应用高斯注意力
        gaussian_out = self.gaussian_attention(x)  # [B, C]
        # print("gaussian_out==>>",gaussian_out.shape)
        # 为了与 MultiheadAttention 兼容，调整形状
        gaussian_out = gaussian_out.unsqueeze(1)  # [B, 1, C]
        gaussian_out = gaussian_out.repeat(1, x.size(1), 1)  # [B, N, C]，与输入 x 形状匹配

        # 通过多头注意力
        attn_output, _ = self.multihead_attention(x.transpose(0, 1), gaussian_out.transpose(0, 1), gaussian_out.transpose(0, 1))  # [N, B, C]
        
        return attn_output.transpose(0, 1)  # 转换回 [B, N, C]

class ParticleFilter(nn.Module):
    def __init__(self, num_particles, state_dim, observation_dim):
        super(ParticleFilter, self).__init__()
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.observation_dim = observation_dim

    def forward(self, z, observation):
        # 检查输入的维度
        assert z.dim() == 2, f"Expected z to have 2 dimensions, but got {z.dim()} dimensions."
        assert observation.dim() == 3, f"Expected observation to have 3 dimensions, but got {observation.dim()} dimensions."

        # 初始化粒子
        particles = z.unsqueeze(1).expand(-1, self.num_particles, -1)  # [B, num_particles, state_dim]
        weights = torch.ones((z.shape[0], self.num_particles), device=z.device) / self.num_particles  # [B, num_particles]

        pdb.set_trace()
        # 预测步骤
        for t in range(observation.shape[2]):  # 遍历观测时间序列
            # 添加过程噪声
            particles = particles + torch.randn_like(particles) * 0.1  # 添加过程噪声

            # 计算权重
            # 扩展 observation 的维度以匹配 particles
            obs_expanded = observation[:, :, t]  # 取出第 t 个时刻的观测，形状为 [B, observation_dim]
            obs_expanded = obs_expanded.unsqueeze(1)  # 扩展为 [B, 1, observation_dim]
            likelihood = self.compute_likelihood(particles, obs_expanded)  # [B, num_particles]
            weights *= likelihood
            weights += 1e-10  # 防止除零
            weights /= weights.sum(dim=1, keepdim=True)  # 归一化权重

            # 重要性重采样
            indices = torch.multinomial(weights, self.num_particles, replacement=True)  # 重采样
            particles = particles[torch.arange(particles.shape[0]).unsqueeze(1), indices]  # 重采样粒子

        return particles.mean(dim=1)  # 返回粒子均值作为新的状态估计

    def compute_likelihood(self, particles, observation):
        # 计算粒子的似然性
        # 假设观测为高斯分布，观测噪声方差为1
        likelihood = torch.exp(-0.5 * ((particles - observation) ** 2).sum(dim=-1))
        likelihood += 1e-8
        return likelihood


class CVAE(nn.Module):

    def __init__(self, in_dim, z_dim, condition_dim=512, num_classes=40):

        super().__init__()
        self.z_dim = z_dim
        # encoder 
        self.encoder = Encoder(in_dim, z_dim)
        # decoder this is cat
        self.decoder = Decoder(z_dim + condition_dim) # z_dim + c_dim

        # in_dim = 512
        self.embedding = nn.Embedding(num_classes, condition_dim)

    def forward(self, x, condition=None):
        # condition [32, 512] 
        # x [32,3,1024]
        # pdb.set_trace()

        # condition
        B,C,N = x.shape
        # print("condition==>>",condition.shape)
        point_feature, mu, log_var, z = self.encoder(x)

        # condition
        condition = self.embedding(condition)

        z = torch.cat([z, condition], dim=1) ## [128, (1024 + 512)]

        out = self.decoder(z)
        recon_pc = out.view(B, C, N)

        return mu, log_var, z, recon_pc

class Encoder(nn.Module):

    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.MLP1 = nn.Sequential(
            SharedMLP(in_dim, 64)
        )
        self.MLP2 = nn.Sequential(
            SharedMLP(64, 64),
            SharedMLP(64, 128),
            SharedMLP(128, 256),
            SharedMLP(256, 512),
        )
        self.fc_mu = nn.Sequential(
            LinearMLP(512, z_dim),
            nn.Linear(z_dim, z_dim)
        )
        self.fc_var = nn.Sequential(
            LinearMLP(512, z_dim),
            nn.Linear(z_dim, z_dim)
        )

        self.fc_global = nn.Sequential(
            LinearMLP(512, z_dim),
            nn.Linear(z_dim, z_dim)
        )
        in_dim = 512
        # self.denoise_attention = GaussianAttention(in_dim)
        # self.denoise_attention = CombinedAttention(in_dim, 4)

    def forward(self, x):
        device = x.device
        # pdb.set_trace()
        # get point feature
        point_feature = self.MLP1(x) # [128,64,1024]
        # get global feature
        global_feature = self.MLP2(point_feature) # [128,512,1024]

        global_feature = torch.max(global_feature, dim=2)[0] # [128,512]

        # ==================
        # global_feature = self.fc_global(global_feature)
        # return global_feature
        # ==================

        # get mean and variance
        mu = self.fc_mu(global_feature)
        log_var = self.fc_var(global_feature)

        # reparametrization tric
        eps = torch.randn_like(torch.exp(log_var), device=device)
        z = mu + torch.exp(0.5*log_var)*eps

        return point_feature, mu, log_var, z


class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        # self.SharedMLP = nn.Sequential(
        #     # SharedMLP(64+z_dim, 256),
        #     # SharedMLP(256, 512),
        #     # SharedMLP(512, 1024),
        #     # SharedMLP(1024, 256),
        #     # SharedMLP(256, 64),
        #     # SharedMLP(64, 3),
        #     # nn.Conv1d(3, 3, 1)
        #     SharedMLP(1+z_dim, 64),
        #     SharedMLP(64, 32),
        #     SharedMLP(32, 16),
        #     SharedMLP(16, 3),
        #     nn.Conv1d(3, 3, 1)
        # )
        self.fc = nn.Sequential(
            LinearMLP(z_dim, 128),
            LinearMLP(128, 256),
            LinearMLP(256, 512),
            LinearMLP(512, 1024),
            LinearMLP(1024, 1024*3),
            nn.Linear(1024*3, 1024*3)
            # LinearMLP(1024, 6000),
            # nn.Linear(6000, 6000)
        )

    def forward(self, x):
        out = self.fc(x)
        return out


class SharedMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        out = self.main(x)
        return out

class LinearMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        out = self.main(x)
        return out


if __name__ == "__main__":
    pass

