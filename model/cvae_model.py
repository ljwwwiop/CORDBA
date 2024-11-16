import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Codebook for quantization
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, z):
        # Flatten the input
        z_flattened = z.view(-1, self.embedding_dim)

        # Compute distances between z and embeddings
        distances = (torch.sum(z_flattened ** 2, dim=1, keepdim=True) 
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(z_flattened, self.embeddings.weight.t()))
        
        # Get the closest embedding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embeddings(encoding_indices).view(z.shape)

        # Compute commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Add the quantization noise in the gradient path
        quantized = z + (quantized - z).detach()

        return quantized, loss


class GumbelQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, temperature=1.0, commitment_cost=0.25):
        super(GumbelQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.temperature = temperature
        self.commitment_cost = commitment_cost

        # Codebook for quantization (embedding vectors)
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward_old(self, z):
        # Flatten the input
        z_flattened = z.view(-1, self.embedding_dim)

        # Compute distances between z and embeddings
        # Similar to how VectorQuantizer works, we calculate distances to embeddings
        distances = (torch.sum(z_flattened ** 2, dim=1, keepdim=True) 
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(z_flattened, self.embeddings.weight.t()))

        # Gumbel-Softmax sampling
        # Compute the logits (the raw distance values between the input z and embeddings)
        logits = -distances
        
        # Add Gumbel noise to the logits
        gumbel_noise = torch.rand_like(logits).log().neg().add_(1)
        logits = logits + gumbel_noise

        # Apply softmax with temperature
        prob = F.softmax(logits / self.temperature, dim=1)
        
        # Sample using Gumbel-Softmax
        quantized = torch.matmul(prob, self.embeddings.weight)
        quantized = quantized.view(z.shape)

        # Compute commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Add the quantization noise in the gradient path
        quantized = z + (quantized - z).detach()

        return quantized, loss

    def forward(self, z):
        # Flatten the input
        # print("z===>>",z.shape)
        z_flattened = z.view(-1, self.embedding_dim)

        # Compute distances between z and embeddings
        # Similar to how VectorQuantizer works, we calculate distances to embeddings
        distances = (torch.sum(z_flattened ** 2, dim=1, keepdim=True) 
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(z_flattened, self.embeddings.weight.t()))

        # Gumbel-Softmax sampling
        # Compute the logits (the raw distance values between the input z and embeddings)
        logits = -distances  # We want to minimize the distance, so the logits should be negative

        # Apply Gumbel-Softmax to get probabilities
        gumbel_sample = F.gumbel_softmax(logits, tau=self.temperature, hard=False)

        # Use softmax results to get quantized embeddings
        quantized = torch.matmul(gumbel_sample, self.embeddings.weight)
        quantized = quantized.view(z.shape)

        # Compute commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Add the quantization noise in the gradient path
        quantized = z + (quantized - z).detach()

        return quantized, loss

class CVAE(nn.Module):

    def __init__(self, in_dim, z_dim, condition_dim=512, num_classes=40):

        super().__init__()
        self.z_dim = z_dim
        # encoder 
        self.encoder = Encoder(in_dim, z_dim)
        # decoder
        self.decoder = Decoder(z_dim + condition_dim)
        # self.decoder = Decoder(z_dim)

        self.embedding = nn.Embedding(num_classes, condition_dim)

        # quantizer
        num_embeddings = 1024 # 512, 1024
        # num_embeddings = 512 ## shapenet
        # self.vector_quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=z_dim, commitment_cost=1.0)
        self.vector_quantizer = GumbelQuantizer(num_embeddings=num_embeddings, embedding_dim=z_dim, commitment_cost=1.0) ## tmux vae


    def forward(self, x, condition=None):
        # condition [32, 512] 
        # x [32,3,1024]
        # pdb.set_trace()

        # condition
        B,C,N = x.shape

        # point_feature, mu, log_var, z = self.encoder(x)

        point_feature, z = self.encoder(x)

        ## quan
        z, quantization_loss = self.vector_quantizer(z)

        # condition
        condition = self.embedding(condition)
        z = torch.cat([z, condition], dim=1)
        out = self.decoder(z)
        recon_pc = out.view(B, C, N)

        # return mu, log_var, z, recon_pc, quantization_loss
        return z, z, z, recon_pc, quantization_loss

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

        # 1D convolution to replace max-pooling (to get global features)
        self.conv1d = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        # self.conv2d = nn.Conv1d(512, z_dim, kernel_size=3, stride=1, padding=1)

        # Fully connected layer to map to z (latent vector)
        # self.fc = nn.Linear(z_dim, z_dim)

    def forward(self, x):
        device = x.device

        # get point feature
        point_feature = self.MLP1(x)
        # get global feature
        global_feature = self.MLP2(point_feature) # [128, 512, 1024]

        # global_feature = torch.max(global_feature, dim=2)[0]
        # ==================
        # global_feature = self.fc_global(global_feature)
        # return global_feature
        # ==================
        
        # get mean and variance
        # mu = self.fc_mu(global_feature)
        # log_var = self.fc_var(global_feature)

        # # reparametrization tric
        # eps = torch.randn_like(torch.exp(log_var), device=device)
        # z = mu + torch.exp(0.5*log_var)*eps

        # return point_feature, mu, log_var, z

        ## vqvae is not vari

        # Apply 1D convolutions to extract global feature representation
        global_feature = self.conv1d(global_feature)  # [batch_size, 256, num_points]
        global_feature = F.relu(global_feature)
        global_feature = global_feature.mean(dim=2)  # Global pooling over the points
        z = self.fc_global(global_feature)

        return point_feature, z


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

