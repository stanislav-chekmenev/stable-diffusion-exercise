import math
import torch

from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        '''
        Args:
            n_heads: Number of heads
            d_embed: Embedding dimension (number of channels a token/pixel is encoded into)
            in_proj_bias: Whether to include bias in the linear transformation
            out_proj_bias: Whether to include bias in the linear transformation
        '''
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        '''
        Args:
            x: latent: (Batch_size, Seq_length, Dim)
            causal_mask: Whether to apply a mask to the attention matrix
            
            Returns: (Batch_size, Seq_length, Dim) - dynamic weighted average (with attention weights) of the
            input sequence embeddings, i.e. the self-attention mechanism output applied to each token/pixel in the input
        '''
        
        input_shape = x.shape
        batch_size, seq_length, _ = input_shape

        interim_shape = (batch_size, seq_length, self.n_heads, self.d_head)

        # (Batch_size, Seq_length, Dim) -> (Batch_size, Seq_length, 3 * Dim) -> 
        # -> 3 tensors of shape (Batch_size, Seq_length, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (Batch_size, Seq_length, Dim) -> (Batch_size, n_heads, Seq_length, d_head)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_size, n_heads, Seq_length, d_head) @ (Batch_size, n_heads, d_head. Seq_length) ->
        # -> (Batch_size, n_heads, Seq_length, Seq_length)
        weight = q @ k.transpose(-1, -2)

        # Apply the mask -> (Batch_size, n_heads, Seq_length, Seq_length)
        if causal_mask:
            # Mask where the upper triangle (higher than the main diagonala) is made up of ones
            # This mask simulates the causality of the attention mechanism where each new token/pixel
            # can only attend to the previous tokens/pixels in the sequence
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        # Normalize the weights to avvoid explosion of the variance
        weight /= math.sqrt(self.d_head)

        # Take softmax to calculate the actual normalized scores
        weight = F.softmax(weight, dim=-1)

        # (Batch_size, n_heads, Seq_length, Seq_length) @ (Batch_size, n_heads, Seq_length, d_head) ->
        # -> (Batch_size, n_heads, Seq_length, d_head)
        # Compute weighted average of the values
        output = weight @ v

        # (Batch_size, n_heads, Seq_length, d_head) -> (Batch_size, Seq_length, n_heads, d_head)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        # (Batch_size, Seq_length, Dim) -> (Batch_size, Seq_length, Dim)
        output = self.out_proj(output, bias=self.out_proj)

        return output
    

class CrossAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        '''
        Args:
            n_heads: Number of heads
            d_embed: Embedding dimension (number of channels a pixel is encoded into)
            d_cross: Dimension of the context (number of channels a token is encoded into)
            in_proj_bias: Whether to include bias in the linear transformation
            out_proj_bias: Whether to include bias in the linear transformation
        '''
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        '''
        Args:
            x: latent: (Batch_size, Seq_len_Q, Dim_Q)
            y: context: (Batch_size, Seq_len_KV, Dim_KV) = (Batch_size, 77, 768)
        '''

        input_shape = x.shape
        batch_size, _, _ = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Multiply Q by Wq, K by Wk, V by Wv
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_size, n_heads, Seq_len_Q, d_head) @ (Batch_size, n_heads, d_head, Seq_len_KV) ->
        # -> (Batch_size, n_heads, Seq_len_Q, Seq_len_KV)
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)

        # No causal mask because we relate all pixels to all tokens
        weight = F.softmax(weight, dim=-1)

        # (Batch_size, n_heads, Seq_len_Q, Seq_len_KV) @ (Batch_size, n_heads, Seq_len_KV, d_head) ->
        # -> (Batch_size, n_heads, Seq_len_Q, d_head)
        # Compute weighted average of the values
        output = weight @ v

        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)
        
        return output


