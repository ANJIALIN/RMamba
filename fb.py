import torch
import torch.nn as nn
from thop import profile
import torch.nn.functional as F

class LightweightPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_dim=64, img_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Patch Embedding
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

        # Learnable Time Token
        self.time_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        # Positional Embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim))

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)

        time_tok = self.time_token.expand(B, -1, -1)
        x = torch.cat([time_tok, x], dim=1)

        x = x + self.pos_embedding
        return x

if __name__ == "__main__":

    # Model initialization and profiling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightPatchEmbed(in_channels=3, patch_size=16, emb_dim=4096).to(device)

    # Generate a sample input tensor for profiling
    sample_input = torch.randn(1, 3, 256, 256).to(device)
    flops, params = profile(model, inputs=(sample_input,))
    print(f'MFLOPs: {flops / 1e6:.2f} M')
    print(f'Parameters: {params / 1e6:.2f} M')

    # Model inference
    output = model(sample_input)
    print(f"Output Shape: {output.shape}")
