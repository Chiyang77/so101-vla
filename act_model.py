"""
Minimal ACT (Action Chunking Transformer) — state-based version.

Architecture (three transformer pieces):

  1. CVAE encoder ("style encoder") — TRAINING ONLY
     Inputs:  [CLS] + obs + the expert's actual K-action chunk
     Output:  a latent z (mu, logvar) summarizing the "intent/style" of
              that chunk. This is what disambiguates multimodal actions:
              two demos with the same obs but different actions get
              different z, so the decoder never has to average them.

  2. Main encoder
     Inputs:  obs + z  (as tokens)
     Output:  "memory" the decoder attends to.

  3. Decoder
     Inputs:  K learned query tokens (one per future timestep)
     Output:  K actions (the chunk), via cross-attention to memory.

At TEST time we skip the CVAE encoder and set z = 0 (prior mean), which
yields one coherent action chunk instead of a per-step average.

Loss = L1(predicted chunk, expert chunk) + beta * KL(z || N(0,1)).
"""

import torch
import torch.nn as nn


class ACTPolicy(nn.Module):
    def __init__(self, obs_dim=18, action_dim=6, chunk_size=50,
                 z_dim=32, d_model=256, nhead=8,
                 enc_layers=4, dec_layers=4, dim_ff=512, dropout=0.1):
        super().__init__()
        self.K = chunk_size
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.d_model = d_model

        # ---- 1. CVAE encoder (style encoder), used only in training ----
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cvae_obs_proj = nn.Linear(obs_dim, d_model)
        self.cvae_act_proj = nn.Linear(action_dim, d_model)
        # positional embedding for [CLS, obs, a_1..a_K]  -> length K+2
        self.cvae_pos = nn.Parameter(torch.zeros(1, chunk_size + 2, d_model))
        cvae_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True)
        self.cvae_encoder = nn.TransformerEncoder(cvae_layer, enc_layers)
        self.to_latent = nn.Linear(d_model, 2 * z_dim)   # -> (mu, logvar)

        # ---- 2. Main encoder (obs + z -> memory) ----
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.z_proj   = nn.Linear(z_dim, d_model)
        self.enc_pos  = nn.Parameter(torch.zeros(1, 2, d_model))  # 2 tokens: obs, z
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, enc_layers)

        # ---- 3. Decoder (K queries -> K actions) ----
        self.query_embed = nn.Embedding(chunk_size, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
        self.action_head = nn.Linear(d_model, action_dim)

        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.cvae_pos, std=0.02)
        nn.init.normal_(self.enc_pos, std=0.02)

    # ------------------------------------------------------------------
    def encode_style(self, obs, actions):
        """CVAE encoder: (obs, action chunk) -> (mu, logvar)."""
        B = obs.shape[0]
        cls = self.cls_token.expand(B, 1, self.d_model)
        obs_tok = self.cvae_obs_proj(obs).unsqueeze(1)        # (B,1,d)
        act_tok = self.cvae_act_proj(actions)                 # (B,K,d)
        seq = torch.cat([cls, obs_tok, act_tok], dim=1)       # (B,K+2,d)
        seq = seq + self.cvae_pos[:, : seq.shape[1]]
        h = self.cvae_encoder(seq)[:, 0]                      # take CLS output
        mu, logvar = self.to_latent(h).chunk(2, dim=-1)
        return mu, logvar

    def decode(self, obs, z):
        """Main encoder + decoder: (obs, z) -> predicted K-action chunk."""
        B = obs.shape[0]
        obs_tok = self.obs_proj(obs).unsqueeze(1)             # (B,1,d)
        z_tok   = self.z_proj(z).unsqueeze(1)                 # (B,1,d)
        memory_in = torch.cat([obs_tok, z_tok], dim=1) + self.enc_pos
        memory = self.encoder(memory_in)                      # (B,2,d)

        queries = self.query_embed.weight.unsqueeze(0).expand(B, self.K, self.d_model)
        dec = self.decoder(tgt=queries, memory=memory)        # (B,K,d)
        return self.action_head(dec)                          # (B,K,action_dim)

    def forward(self, obs, actions=None):
        """
        Training (actions given): use CVAE encoder -> z (reparam) -> decode.
        Inference (actions None) : z = 0 -> decode.
        Returns (pred_actions, mu, logvar). mu/logvar are None at inference.
        """
        if actions is not None:
            mu, logvar = self.encode_style(obs, actions)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)              # reparameterization
        else:
            mu = logvar = None
            z = torch.zeros(obs.shape[0], self.z_dim, device=obs.device)

        pred = self.decode(obs, z)
        return pred, mu, logvar


def kl_divergence(mu, logvar):
    """KL( N(mu, sigma) || N(0,1) ), averaged over batch."""
    # -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) per sample, then mean
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()
