"""Primary Wearformer architecture, extracted from the experimental runner.

Parameter names and initialization order match the primary ``TimeGapSAITS``
configuration. The unused second-decoder parameters are retained for strict
loading of its checkpoints. Historical experimental model variants are omitted.
SAITS-derived attention and feed-forward operations are identified in
THIRD_PARTY.md.
"""

import math
from typing import Optional

import torch
from torch import nn


class OfficialSinusoidalPositionEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                             * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(seq_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[:encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, x):
        return x + self.encoding[:, :x.shape[1]].to(dtype=x.dtype)


class OfficialDiagonalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, attn_dropout):
        super().__init__()
        if d_model != n_heads * d_k:
            raise ValueError("d_model must equal n_heads * d_k")
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.query = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.key = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.value = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.output = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.attention_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, diagonal_mask):
        batch, seq_len, _ = x.shape
        query = self.query(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        key = self.key(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        value = self.value(x).view(batch, seq_len, self.n_heads, self.d_v).transpose(1, 2)
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
        scores = scores / math.sqrt(self.d_k)
        scores = scores.masked_fill(diagonal_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        dropped = self.attention_dropout(attention).to(dtype=value.dtype)
        context = torch.matmul(dropped, value)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.d_v)
        return self.output(context), dropped.float().mean(dim=1)


class LocalConvBranch(nn.Module):
    """Residual depthwise temporal convolution with circular daily padding."""
    def __init__(self, d_model, kernel=9, dropout=0.0):
        super().__init__()
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("local_conv must be a positive odd number")
        self.norm = nn.LayerNorm(d_model)
        self.dw = nn.Conv1d(d_model, d_model, kernel, padding=kernel // 2,
                            groups=d_model, padding_mode="circular")
        self.pw = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

    def forward(self, x):
        h = self.dw(self.norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.dropout(self.pw(torch.relu(h)))


class OfficialPreLNPositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.linear_1 = nn.Linear(d_model, d_ffn)
        self.linear_2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        transformed = self.layer_norm(x)
        transformed = self.linear_2(torch.relu(self.linear_1(transformed)))
        return x + self.dropout(transformed)


class OfficialPreLNEncoderLayer(nn.Module):
    def __init__(self, seq_len, d_model, n_heads, d_k, d_v, d_ffn,
                 dropout, attn_dropout, local_conv):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = OfficialDiagonalMultiHeadAttention(d_model, n_heads, d_k, d_v, attn_dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.local_conv = LocalConvBranch(d_model, local_conv, dropout) if local_conv else None
        self.feed_forward = OfficialPreLNPositionWiseFFN(d_model, d_ffn, dropout)
        # Primary Wearformer permits attention to the current position.
        self.register_buffer("diagonal_attention_mask",
                             torch.zeros(1, 1, seq_len, seq_len, dtype=torch.bool), persistent=False)

    def forward(self, x):
        attended, attention = self.attention(self.attention_norm(x), self.diagonal_attention_mask)
        x = x + self.attention_dropout(attended)
        if self.local_conv is not None:
            x = self.local_conv(x)
        return self.feed_forward(x), attention


class OfficialPreLNDMSABlock(nn.Module):
    def __init__(self, n_layers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([OfficialPreLNEncoderLayer(**kwargs) for _ in range(n_layers)])

    def forward(self, x):
        attention = None
        for layer in self.layers:
            x, attention = layer(x)
        return x, attention


class TimeGapSAITS(nn.Module):
    """Wearformer with daily/weekly time, mask distances and eight personal features.

    Inputs have shapes ``values, observed: [batch, time, 2]`` and
    ``side_information: [batch, time, 16]``. Missing input values must already be
    set to zero in standardized units. The dictionary output retains the names
    used by the training implementation and the uncertainty head.
    """
    def __init__(self, seq_len: int, d_model: int = 256, n_features: int = 2,
                 n_layers: int = 2, n_heads: int = 4, d_k: int = 64,
                 d_v: int = 64, d_ffn: int = 256, dropout: float = 0.0,
                 attn_dropout: float = 0.0, local_conv: int = 9,
                 side_information_mode: str = "time_gap_profile",
                 gap_feature_count: Optional[int] = None, profile_feature_count: int = 8,
                 single_block: bool = True, no_diagonal_mask: bool = True):
        super().__init__()
        if n_features != 2 or profile_feature_count != 8 or gap_feature_count not in (None, 4):
            raise ValueError("Primary Wearformer requires two signals, four distances and eight personal features")
        if not single_block or not no_diagonal_mask or side_information_mode != "time_gap_profile":
            raise ValueError("Only the primary single-stack Wearformer configuration is supported")
        if d_model != n_heads * d_k or seq_len < 1 or n_layers < 1:
            raise ValueError("Invalid sequence length, layer count or attention dimensions")
        self.seq_len, self.n_features = seq_len, n_features
        self.day_len, self.context_days = seq_len, 0
        self.window_revin, self.single_block = False, True
        self.side_information_mode = side_information_mode
        self.time_feature_count, self.gap_feature_count, self.profile_feature_count = 4, 4, 8
        # Keep construction order identical to the frozen primary model.
        self.embedding_1 = nn.Linear(n_features * 2, d_model)
        self.embedding_2 = nn.Linear(n_features * 2, d_model)
        self.position_encoding = OfficialSinusoidalPositionEncoding(seq_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.dmsa_block_1 = OfficialPreLNDMSABlock(
            n_layers=n_layers, seq_len=seq_len, d_model=d_model, n_heads=n_heads,
            d_k=d_k, d_v=d_v, d_ffn=d_ffn, dropout=dropout,
            attn_dropout=attn_dropout, local_conv=local_conv)
        self.dmsa_block_2 = None
        self.reduce_dim_z = nn.Linear(d_model, n_features)
        self.reduce_dim_beta = nn.Linear(d_model, n_features)
        self.reduce_dim_gamma = nn.Linear(n_features, n_features)
        self.weight_combine = nn.Linear(seq_len + n_features, n_features)
        self.time_embedding = nn.Linear(4, d_model, bias=False)
        self.gap_distance_embedding = nn.Sequential(nn.Linear(4, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.profile_embedding = nn.Sequential(nn.Linear(8, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def encode_first(self, values, observed, side_information):
        if values.ndim != 3 or tuple(values.shape[1:]) != (self.seq_len, 2):
            raise ValueError(f"Expected values [batch,{self.seq_len},2]")
        if observed.shape != values.shape or side_information.shape != (*values.shape[:2], 16):
            raise ValueError("Input mask or side-information shape mismatch")
        mask = observed.to(dtype=values.dtype)
        side = side_information.to(dtype=values.dtype)
        embedded = self.position_encoding(self.embedding_1(torch.cat((values, mask), dim=-1)))
        embedded = embedded + self.time_embedding(side[..., :4])
        embedded = embedded + self.gap_distance_embedding(side[..., 4:8])
        embedded = embedded + self.profile_embedding(side[..., 8:])
        encoded, _ = self.dmsa_block_1(self.embedding_dropout(embedded))
        return encoded

    def forward(self, values, observed, side_information):
        encoded = self.encode_first(values, observed, side_information)
        prediction = self.reduce_dim_z(encoded)
        mask = observed.to(dtype=values.dtype)
        return {"x_tilde_1": prediction, "x_tilde_2": prediction, "x_tilde_3": prediction,
                "imputed": mask * values + (1.0 - mask) * prediction,
                "encoded_1": encoded, "encoded_2": encoded}
