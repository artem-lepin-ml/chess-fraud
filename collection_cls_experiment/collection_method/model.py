import torch
import torch.nn as nn
from consts import (
    EMB_DIM,
    ENCODER_HIDDEN_DIM,
    NUM_ATTN_HEADS,
    NUM_ENCODER_LAYERS,
    LAST_LAYER_DIM)


class ChessEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(
            in_features=EMB_DIM,
            out_features=ENCODER_HIDDEN_DIM
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, ENCODER_HIDDEN_DIM))
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=ENCODER_HIDDEN_DIM,
            nhead=NUM_ATTN_HEADS,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=NUM_ENCODER_LAYERS
        )
        self.move_cls_head = nn.Sequential(
            nn.Linear(ENCODER_HIDDEN_DIM, LAST_LAYER_DIM),
            nn.GELU(),
            nn.Linear(LAST_LAYER_DIM, 1)
        )
        self.game_cls_head = nn.Sequential(
            nn.Linear(ENCODER_HIDDEN_DIM, LAST_LAYER_DIM),
            nn.GELU(),
            nn.Linear(LAST_LAYER_DIM, 1)
        )

    def forward(self, embeddings, padding_mask):
        B, S, E = embeddings.shape

        cls_tokens = self.cls_token.expand(B, 1, -1)
        outputs = self.linear(embeddings)
        outputs = torch.cat([cls_tokens, outputs], dim=1)
        outputs = self.transformer_encoder(
            outputs,
            src_key_padding_mask=padding_mask
        )

        move_outputs = self.move_cls_head(outputs[:, 1:, :])
        game_outputs = self.game_cls_head(outputs[:, 0, :])
        return move_outputs, game_outputs
