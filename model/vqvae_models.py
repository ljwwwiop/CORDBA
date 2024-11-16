

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

class VQVAE_Pointnet(nn.Module):

    def __init__(self, encoder, decoder, in_dim, z_dim, condition_dim=512, num_classes=40):

        super().__init__()
        self.z_dim = z_dim
        # encoder 
        self.encoder = encoder
        # decoder
        self.decoder = decoder
        # self.decoder = Decoder(z_dim)

        self.embedding = nn.Embedding(num_classes, condition_dim)

        # quantizer
        num_embeddings = 512 # 512, 1024
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

        # point_feature, z = self.encoder(x)
        z, _, _ = self.encoder(x)

        ## quan
        z, quantization_loss = self.vector_quantizer(z)

        # condition
        condition = self.embedding(condition)
        z = torch.cat([z, condition], dim=1)

        out = self.decoder(z)
        # recon_pc = out

        recon_pc = out.view(B, C, N)

        # return mu, log_var, z, recon_pc, quantization_loss
        return z, z, z, recon_pc, quantization_loss


class PointCloudDecoderMLP(nn.Module):
    def __init__(self, latent_dim, num_hidden, num_point=2500, point_dim=3, bn_decay=0.5):
        self.num_point = num_point
        self.point_dim = point_dim 
        self.latent_dim = latent_dim

        super(PointCloudDecoderMLP, self).__init__()

        self.fc1 = nn.Linear(self.latent_dim, self.latent_dim*2)
        self.fc2 = nn.Linear(self.latent_dim*2, self.latent_dim*4)
        self.fc3 = nn.Linear(self.latent_dim*4, self.latent_dim*8)
        self.fc4 = nn.Linear(self.latent_dim*8, self.latent_dim*16)
        self.fc5 = nn.Linear(self.latent_dim*16, self.latent_dim*32)
        self.fc6 = nn.Linear(self.latent_dim*32, self.latent_dim*32)
        self.fcend = nn.Linear(self.latent_dim*32, int(self.point_dim*self.num_point))

    def forward(self, x):
        # UPCONV Decoder
        #x = x.view(x.size(0), self.latent_dim, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        x = nn.functional.relu(self.fc5(x))
        x = nn.functional.relu(self.fc6(x))
        x = self.fcend(x)
        x = x.reshape(-1, self.num_point, self.point_dim)
        return x





if __name__ == "__main__":
    pass

