import torch
import torch.nn as nn
from utils.config_utils import ModelConfig


class ChessEncoder(nn.Module):
    """
    Transformer-based encoder for chess move collections.

    Takes move embeddings and produces both move-level and game-level
    predictions for cheating detection.

    Architecture:
    - Linear projection from embedding dimension to hidden dimension
    - CLS token for game-level classification
    - Transformer encoder layers
    - Separate classification heads for moves and games
    """

    def __init__(self, model_cfg: ModelConfig):
        super().__init__()

        self.embedding_dim = model_cfg.embedding_dim
        self.encoder_hidden_dim = model_cfg.encoder_hidden_dim
        self.num_attention_heads = model_cfg.num_attention_heads
        self.num_encoder_layers = model_cfg.num_encoder_layers
        self.last_layer_dim = model_cfg.last_layer_dim

        # Project embeddings to hidden dimension
        self.linear = nn.Linear(self.embedding_dim, self.encoder_hidden_dim)

        # Learnable CLS token for game-level classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.encoder_hidden_dim))

        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_hidden_dim,
            nhead=self.num_attention_heads,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=self.num_encoder_layers,
        )

        # Move-level classification head
        self.move_cls_head = nn.Sequential(
            nn.Linear(self.encoder_hidden_dim, self.last_layer_dim),
            nn.GELU(),
            nn.Linear(self.last_layer_dim, 1),
        )

        # Game-level classification head (uses CLS token)
        self.game_cls_head = nn.Sequential(
            nn.Linear(self.encoder_hidden_dim, self.last_layer_dim),
            nn.GELU(),
            nn.Linear(self.last_layer_dim, 1),
        )

    def forward(self, embeddings, padding_mask):
        """
        Forward pass through the Chess Encoder.

        Args:
            embeddings: Tensor of shape (B, S, E) where B=batch, S=seq, E=emb_dim
            padding_mask: Boolean tensor of shape (B, S+1) where True indicates padding

        Returns:
            move_outputs: Tensor of shape (B, S, 1) - move-level predictions
            game_outputs: Tensor of shape (B, 1) - game-level predictions
        """
        B, S, E = embeddings.shape

        # Add CLS token at the beginning of the sequence
        cls_tokens = self.cls_token.expand(B, 1, -1)
        outputs = self.linear(embeddings)
        outputs = torch.cat([cls_tokens, outputs], dim=1)

        # Apply transformer encoder
        outputs = self.transformer_encoder(outputs, src_key_padding_mask=padding_mask)

        # Get predictions
        move_outputs = self.move_cls_head(outputs[:, 1:, :])  # Skip CLS token
        game_outputs = self.game_cls_head(outputs[:, 0, :])   # Only CLS token

        return move_outputs, game_outputs
