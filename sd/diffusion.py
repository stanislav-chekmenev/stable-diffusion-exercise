import torch

from torch import nn
from torch.nn import functional as F

from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):

    def __init__(self, d_embed: int):
        super().__init__()

        self.linear_1 = nn.Linear(d_embed, 4 * d_embed)
        self.linear_2 = nn.Linear(4 * d_embed, 4 * d_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: (Batch_size, 320)
        '''
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)

        # (Batch_size, 1280)
        return x


class UNet_ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, d_time=1280):
        super().__init__()

        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(d_time, out_channels)
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature: torch.Tensor, time: torch.Tensor):
        '''
        Args:
            feature: (Batch_size, Channels, Height, Width)
            time: (1, 1280)
        '''
        residue = feature

        feature = self.groupnorm_feature(feature)

        feature = F.silu(feature)

        feature = self.conv_feature(feature)

        time = self.linear_time(time)

        time = F.silu(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)
    

class UNet_AttentionBlock(nn.Module):

    def __init__(self, n_head: int, d_embed: int, d_context: int=768):
        super().__init__()
        channels = n_head * d_embed

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)

        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_gelu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_gelu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        '''
        Args:
            x: (Batch_size, Channels, Height, Width)
            context: (Batch_size, Seq_length, Dim=768)
        '''

        residue_long = x 

        x = self.groupnorm(x)

        x = self.conv_input(x)

        n, c, h, w = x.shape

        # (Batch_size, Channels, Height, Width) -> (Batch_size, Channels, Height * Width)
        x = x.view((n, c, h * w))

        # (Batch_size, Channels, Height * Width) -> (Batch_size, Height * Width, Channels)
        x = x.transpose(-1, -2)

        # Normalization + Self-Attention with skip connection
        residue_short = x

        x = self.layernorm_1(x)
        x = self.attention_1(x)

        x += residue_short

        residue_short = x

        # Normalization + Cross-Attention with skip connection
        x = self.layernorm_2(x)

        # Cross attention mechanism
        x = self.attention_2(x, context)

        x += residue_short

        residue_short = x

        # Normalization + Feedforward with GeLU and skip connection
        x = self.layernorm_3(x)

        x, gate = self.linear_gelu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_gelu_2(x)

        x += residue_short

        # (Batch_size, Height * Width, Channels) -> (Batch_size, Channels, Height * Width)
        x = x.transpose(-1, -2)

        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long

    




class UpSample(nn.Module):

    def __init__(self, channels: int):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x: (Batch_size, Channels, Height, Width)
        '''
        # (Batch_size, Channels, Height, Width) -> (Batch_size, Channels, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SwitchSequential(nn.Sequential):

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        for layer in self:
            if isinstance(layer, UNet_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNet_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        return x        



class UNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.encoder = nn.ModuleList([
            # (Batch_size, 4, Height / 8, Width / 8) -> (Batch_size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),

            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),

            # (Batch_size, 320, Height / 8, Width / 8) -> (Batch_size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNet_ResidualBlock(320, 640), UNet_AttentionBlock(8, 80)),

            SwitchSequential(UNet_ResidualBlock(640, 640), UNet_AttentionBlock(8, 80)),

            # (Batch_size, 640, Height / 16, Width / 16) -> (Batch_size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNet_ResidualBlock(640, 1280), UNet_AttentionBlock(8, 160)),

            SwitchSequential(UNet_ResidualBlock(1280, 1280), UNet_AttentionBlock(8, 160)),

            # (Batch_size, 1280, Height / 32, Width / 32) -> (Batch_size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNet_ResidualBlock(1280, 1280)),

            SwitchSequential(UNet_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNet_ResidualBlock(1280, 1280),

            UNet_AttentionBlock(8, 160),

            UNet_ResidualBlock(1280, 1280),
        )

        self.decoder = nn.ModuleList([
            # The input channels are doubled because of the skip connections
            # (Batch_size, 2560, Height / 64, Width / 64) -> (Batch_size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNet_ResidualBlock(2560, 1280)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280), UpSample(1280)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),

            SwitchSequential(UNet_ResidualBlock(1920, 1280), UNet_AttentionBlock(8, 160), UpSample(1280)),

            SwitchSequential(UNet_ResidualBlock(1920, 640), UNet_AttentionBlock(8, 80)),

            SwitchSequential(UNet_ResidualBlock(1920, 640), UNet_AttentionBlock(8, 80)),

            SwitchSequential(UNet_ResidualBlock(960, 640), UNet_AttentionBlock(8, 80), UpSample(640)),

            SwitchSequential(UNet_ResidualBlock(960, 320), UNet_AttentionBlock(8, 40)),

            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)),
        ])
    

class UNet_Outputlayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: (Batch_size, 320, Height / 8, Width / 8)
        '''
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)

        # (Batch_size, 4, Height / 8, Width / 8)
        return x    


class Diffusion(nn.Module):

    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNet_Outputlayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            latent: (Batch_size, 4, Height / 8, Width / 8),
            context: (Batch_size, Seq_length, Dim=768)
            time: (1, 320)
        
        '''
        # Similar to a positional encoding of a transformer language model
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (Batch_size, 4, Height / 8, Width / 8) -> (Batch_size, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)

        # (Batch_size, 320, Height / 8, Width / 8) -> (Batch_size, 4, Height / 8, Width / 8)
        output = self.final(output)

        # (Batch_size, 4, Height / 8, Width / 8)
        return output
    

