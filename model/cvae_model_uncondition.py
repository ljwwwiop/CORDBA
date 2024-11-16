import os

import torch
import torch.nn as nn
import pdb


## uncondition
class CVAE(nn.Module):

    def __init__(self, in_dim, z_dim, condition_dim=512, num_classes=40):

        super().__init__()
        self.z_dim = z_dim
        # encoder 
        self.encoder = Encoder(in_dim, z_dim)
        # decoder
        self.decoder = Decoder(z_dim)

        # self.embedding = nn.Embedding(num_classes, condition_dim)


    def forward(self, x, condition=None):
        # condition [32, 512] 
        # x [32,3,1024]
        # pdb.set_trace()

        # condition
        B,C,N = x.shape

        point_feature, mu, log_var, z = self.encoder(x)

        # uncondition
        # condition = self.embedding(condition)
        # z = torch.cat([z, condition], dim=1)

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

    def forward(self, x):
        device = x.device

        # get point feature
        point_feature = self.MLP1(x)

        # get global feature
        global_feature = self.MLP2(point_feature)
        global_feature = torch.max(global_feature, dim=2)[0]

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

