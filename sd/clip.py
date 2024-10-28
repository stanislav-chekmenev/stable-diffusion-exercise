import torch

from torch import nn

from attention import SelfAttention


class CLIPEmbedding(nn.Module):

    def __init__(self, n_vocab: int, d_embed: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, d_embed)
        # A learnable weight matrix encodes the position information for each token
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, d_embed))

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # (Batch_size, Seq_length) -> (Batch_size, Seq_length, Dim)
        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x


class CLIPLayer(nn.Module):

    def __init__(self, n_head: int, d_embed: int):
        super().__init__()

        # Pre-attention norm
        self.layernorm_1 = nn.LayerNorm(d_embed)
        # Self-attention
        self.attention = SelfAttention(n_head, d_embed)
        # Pre FNN norm
        self.layernorm_2 = nn.LayerNorm(d_embed)
        # Feedforward neural network
        self.linear_1 = nn.Linear(d_embed, 4 * d_embed)
        self.linear_2 = nn.Linear(4 * d_embed, d_embed)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: (Batch_size, Seq_length, Dim)

        """

        residue = x

        # SELF-ATTENTION
        # TODO: This layernorm might be in the wrong place, check it!
        x = self.layernorm_1(x)

        x = self.attention(x, causal_mask=True)

        x += residue

        # FEEDFORWARD

        residue = x

        x = self.layernorm_2(x)

        x = self.linear_1(x)

        # Apply QuickGELU activation because it's better in practice
        x = x * torch.sigmoid(1.702 * x)

        x = self.linear_2(x)

        x += residue

        return x


class CLIP(nn.Module):

    def __init__(self):
        super().__init__()
        # 77 is the max_seq_length of the embeddings
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([CLIPLayer(12, 768) for _ in range(12)])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (Batch_size, Seq_length) -> (Batch_size, Seq_length, Dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (Batch_size, Seq_length, Dim) -> (Batch_size, Seq_length, Dim)
        output = self.layernorm(state)

        return output
