import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, is_eeg: bool, layer_num=1, peri_dim=0):
        super(TransformerEncoder, self).__init__()
        if is_eeg:
            self.channel_feat_dim = 5
            self.num_channel = 62
        else:
            self.channel_feat_dim = 1
            self.num_channel = peri_dim  # SEED: 33  SEED-IV: 31

        self.positional_encoding = False
        self.hidden_dimension = 64
        self.num_layers = layer_num

        if self.positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model=self.hidden_dimension, dropout=0)
        else:
            self.positional_encoding = None
        self.attn_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dimension, nhead=2, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.attn_layer, num_layers=self.num_layers)
        self.dim_down = nn.Linear(self.num_channel * self.hidden_dimension, self.hidden_dimension)
        self.dim_up = nn.Linear(self.channel_feat_dim, self.hidden_dimension)

    def forward(self, x):
        # (bs,62,5) / (bs,33/31,1)
        assert len(x.shape) == 3 and x.shape[1] == self.num_channel and x.shape[2] == self.channel_feat_dim
        x = self.dim_up(x)  # (bs,62,fusion_hidden) / (bs,33/31,fusion_hidden)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        x = self.encoder(x)  # (bs,62,fusion_hidden) / (bs,33/31,fusion_hidden)
        d_x = self.dim_down(x.view(x.shape[0], -1))  # (bs,fusion_hidden)

        return x, d_x

    def init_weights(self):
        self.dim_up.reset_parameters()
        if self.dim_down is not None:
            self.dim_down.reset_parameters()
        # encoder_layer initialize
        for layer in self.encoder.layers:
            layer.self_attn._reset_parameters()
            layer.linear1.reset_parameters()
            layer.linear2.reset_parameters()
            layer.norm1.reset_parameters()
            layer.norm2.reset_parameters()


class Fusion(nn.Module):
    def __init__(self, peri_dim=0):
        super(Fusion, self).__init__()
        self.backbone_hidden_dimension = 64
        self.fusion_hidden_dimension = 128

        self.eeg_feature_dim_change = nn.Linear(self.backbone_hidden_dimension, self.fusion_hidden_dimension)
        self.eye_feature_dim_change = nn.Linear(self.backbone_hidden_dimension, self.fusion_hidden_dimension)

        self.eeg_eye_attn = nn.MultiheadAttention(embed_dim=self.fusion_hidden_dimension, num_heads=2, batch_first=True)
        self.eye_eeg_attn = nn.MultiheadAttention(embed_dim=self.fusion_hidden_dimension, num_heads=2, batch_first=True)

        self.eeg_feature_dim_down = nn.Linear(62 * self.fusion_hidden_dimension, self.fusion_hidden_dimension)
        self.eye_feature_dim_down = nn.Linear(peri_dim * self.fusion_hidden_dimension, self.fusion_hidden_dimension)

    def forward(self, eeg_spec_data, eye_trac_data):
        # eeg (bs,62,fusion_hidden)
        # peri (bs,33/31,fusion_hidden)
        eeg_spec_data = self.eeg_feature_dim_change(eeg_spec_data)  # (bs,62,h)
        eye_trac_data = self.eye_feature_dim_change(eye_trac_data)  # (bs,33/31,h)

        eeg_attn_out, _ = self.eeg_eye_attn(eeg_spec_data, eye_trac_data, eye_trac_data)  # (bs,62,h)
        eeg_attn_out += eeg_spec_data

        eye_attn_out, _ = self.eye_eeg_attn(eye_trac_data, eeg_spec_data, eeg_spec_data)  # (bs,33/31,h)
        eye_attn_out += eye_trac_data

        eeg_attn_out = eeg_attn_out.contiguous().view(eeg_attn_out.shape[0], -1)
        eeg_out = self.eeg_feature_dim_down(eeg_attn_out)  # (bs,h)

        eye_attn_out = eye_attn_out.contiguous().view(eye_attn_out.shape[0], -1)
        eye_out = self.eye_feature_dim_down(eye_attn_out)  # (bs,h)

        fused_tensor = torch.cat([eeg_out, eye_out], dim=-1)

        return fused_tensor

    def init_weights(self):
        self.eeg_eye_attn._reset_parameters()
        self.eye_eeg_attn._reset_parameters()

        self.eeg_feature_dim_change.reset_parameters()
        self.eye_feature_dim_change.reset_parameters()

        self.eeg_feature_dim_down.reset_parameters()
        self.eye_feature_dim_down.reset_parameters()


network_dict = {'fusion': Fusion,
                'transformer': TransformerEncoder}