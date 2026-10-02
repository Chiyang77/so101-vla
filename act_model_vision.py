"""
Vision ACT: same ACT idea (CVAE + chunking + transformer decoder), but the
observation is a CAMERA IMAGE + proprioception instead of privileged state.

Pieces:
  - VisionEncoder: a small CNN that turns a 128x128 RGB image into a grid of
    feature tokens (4x4 = 16 tokens of d_model). This is how the policy "sees".
  - CVAE encoder (training only): [CLS, proprio, action chunk] -> (mu, logvar).
    (Uses proprio, not the image — it only needs to disambiguate action modes.)
  - Main encoder: [image tokens, proprio token, z token] -> memory.
  - Decoder: K learned queries cross-attend to memory -> K actions.

Inference: z = 0, no CVAE encoder (same as state-based ACT).
"""

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """128x128x3 image -> 16 tokens of d_model (a 4x4 feature grid)."""
    def __init__(self, d_model):
        super().__init__()
        def block(cin, cout, k, s, p):
            return nn.Sequential(
                nn.Conv2d(cin, cout, k, s, p),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.conv = nn.Sequential(
            block(3,   32, 5, 2, 2),   # 128 -> 64
            block(32,  64, 3, 2, 1),   # 64  -> 32
            block(64, 128, 3, 2, 1),   # 32  -> 16
            block(128,256, 3, 2, 1),   # 16  -> 8
            block(256, d_model, 3, 2, 1),  # 8 -> 4
        )
        self.pos = nn.Parameter(torch.zeros(1, 16, d_model))  # 4x4 spatial pos
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, img):                     # img (B,3,128,128) in [0,1]
        f = self.conv(img)                      # (B, d_model, 4, 4)
        tok = f.flatten(2).transpose(1, 2)      # (B, 16, d_model)
        return tok + self.pos


class VisionACTPolicy(nn.Module):
    def __init__(self, proprio_dim=12, action_dim=6, chunk_size=50,
                 z_dim=32, d_model=256, nhead=8,
                 enc_layers=4, dec_layers=4, dim_ff=512, dropout=0.1):
        super().__init__()
        self.K = chunk_size
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.d_model = d_model

        # ---- images: separate encoders for the two views ----
        # external = 3/4 workspace view (locate cube+target across the scene)
        # wrist    = gripper close-up (precise grasp)
        self.img_encoder_ext   = VisionEncoder(d_model)
        self.img_encoder_wrist = VisionEncoder(d_model)
        # per-camera embeddings so the transformer knows which view a token is
        self.cam_ext   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cam_wrist = nn.Parameter(torch.zeros(1, 1, d_model))

        # ---- CVAE encoder (proprio + action chunk), training only ----
        self.cls_token       = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cvae_prop_proj  = nn.Linear(proprio_dim, d_model)
        self.cvae_act_proj   = nn.Linear(action_dim, d_model)
        self.cvae_pos        = nn.Parameter(torch.zeros(1, chunk_size + 2, d_model))
        cvae_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                                batch_first=True)
        self.cvae_encoder = nn.TransformerEncoder(cvae_layer, enc_layers)
        self.to_latent = nn.Linear(d_model, 2 * z_dim)

        # ---- main encoder (image tokens + proprio + z -> memory) ----
        self.prop_proj = nn.Linear(proprio_dim, d_model)
        self.z_proj    = nn.Linear(z_dim, d_model)
        # type embeddings for the proprio and z tokens (image tokens carry their
        # own spatial pos inside VisionEncoder)
        self.prop_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.z_type    = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                               batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, enc_layers)

        # ---- decoder (K queries -> K actions) ----
        self.query_embed = nn.Embedding(chunk_size, d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout,
                                               batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
        self.action_head = nn.Linear(d_model, action_dim)

        for p in [self.cls_token, self.cvae_pos, self.prop_type, self.z_type,
                  self.cam_ext, self.cam_wrist]:
            nn.init.normal_(p, std=0.02)

    # ------------------------------------------------------------------
    def encode_style(self, proprio, actions):
        B = proprio.shape[0]
        cls = self.cls_token.expand(B, 1, self.d_model)
        p_tok = self.cvae_prop_proj(proprio).unsqueeze(1)     # (B,1,d)
        a_tok = self.cvae_act_proj(actions)                   # (B,K,d)
        seq = torch.cat([cls, p_tok, a_tok], dim=1) + self.cvae_pos[:, : self.K + 2]
        h = self.cvae_encoder(seq)[:, 0]
        mu, logvar = self.to_latent(h).chunk(2, dim=-1)
        return mu, logvar

    def decode(self, img_ext, img_wrist, proprio, z):
        B = img_ext.shape[0]
        ext_tok   = self.img_encoder_ext(img_ext)   + self.cam_ext     # (B,16,d)
        wrist_tok = self.img_encoder_wrist(img_wrist) + self.cam_wrist # (B,16,d)
        prop_tok  = self.prop_proj(proprio).unsqueeze(1) + self.prop_type   # (B,1,d)
        z_tok     = self.z_proj(z).unsqueeze(1) + self.z_type               # (B,1,d)
        memory = self.encoder(torch.cat([ext_tok, wrist_tok, prop_tok, z_tok], dim=1))
        queries = self.query_embed.weight.unsqueeze(0).expand(B, self.K, self.d_model)
        dec = self.decoder(tgt=queries, memory=memory)                # (B,K,d)
        return self.action_head(dec)

    def forward(self, img_ext, img_wrist, proprio, actions=None):
        if actions is not None:
            mu, logvar = self.encode_style(proprio, actions)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            mu = logvar = None
            z = torch.zeros(img_ext.shape[0], self.z_dim, device=img_ext.device)
        pred = self.decode(img_ext, img_wrist, proprio, z)
        return pred, mu, logvar


def kl_divergence(mu, logvar):
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()
