import torch
import torch.nn as nn
import numpy as np

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU() # Swish activation
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = self.norm(x)
        h = self.linear1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear2(h)
        return residual + h

class SimpleResNetV2(nn.Module):
    """
    Vector Field Network v(t, x)
    x: [Batch, 3] (Normal Flux, Shear Flux, Collision Count)
    c: [Batch, 3] (RPM, Fill, Ball)
    t: [Batch] (Time scalar)
    """
    def __init__(self, dim_input=3, dim_cond=3, dim_hidden=512, num_layers=8, dropout=0.1):
        super().__init__()
        
        # Time Embedding
        self.time_dim = 128
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_dim),
            nn.Linear(self.time_dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.time_dim),
        )

        # Input Projection
        full_input_dim = dim_input + dim_cond + self.time_dim
        self.input_mlp = nn.Sequential(
            nn.Linear(full_input_dim, dim_hidden),
            nn.SiLU()
        )

        # Residual Layers
        self.layers = nn.ModuleList([
            ResidualBlock(dim_hidden, dropout=dropout) for _ in range(num_layers)
        ])

        # Output Projection
        self.final_layer = nn.Linear(dim_hidden, dim_input)
        
        # Zero-init final layer for flow matching stability
        nn.init.zeros_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, t, x, c):
        # 1. Embed Time
        t_emb = self.time_mlp(t)
        
        # 2. Concat [x, c, t]
        h = torch.cat([x, c, t_emb], dim=-1)
        
        # 3. Process
        h = self.input_mlp(h)
        for layer in self.layers:
            h = layer(h)
            
        # 4. Output Velocity Field
        return self.final_layer(h)

