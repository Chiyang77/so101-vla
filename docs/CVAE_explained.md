# The CVAE in our ACT — explained

This documents the Conditional Variational Auto-Encoder (CVAE) inside
`act_model.py` — what it is, why it's there, and how each piece of the code
implements it. Read alongside `act_model.py`.

---

## 1. The problem it solves: multimodality

Behavior cloning is "given observation `o`, predict the expert's action `a`."
Trained with MSE/L1, the network learns the **average** action for each
observation. That's fine when there's one right answer — but manipulation is
**multimodal**: the same observation can have several equally-valid actions.

> Gripper hovering over the cube: the expert sometimes nudges left, sometimes
> right, sometimes straight down. All are valid. Average them → a mushy
> in-between action that grasps nothing.

Averaging incompatible modes is *the* failure of naive BC. The CVAE fixes it by
giving the network a way to **commit to one mode** instead of averaging.

---

## 2. VAE in one paragraph

An **auto-encoder** squeezes data through a bottleneck: `encoder: x -> z`,
`decoder: z -> x̂`, trained so `x̂ ≈ x`. The latent `z` is a compressed code.

A **Variational** AE makes `z` *probabilistic*: the encoder outputs a
distribution `N(mu, sigma)` instead of a point, and we add a loss term pulling
that distribution toward a standard normal `N(0,1)`. Result: the latent space is
smooth and we can **sample** `z` (or set `z=0`) at test time and get valid
outputs. The two losses:

- **Reconstruction**: decoder output should match the input (here: L1 on actions).
- **KL divergence**: `KL(N(mu,sigma) || N(0,1))` — keep the latent near the prior.

---

## 3. What "Conditional" adds

Plain VAE models `p(x)`. We don't want to generate *any* action chunk — we want
the chunk **for the current observation**. So everything is **conditioned on the
observation `o`**:

- encoder models `q(z | o, a)` — "what intent produced this action chunk, given o?"
- decoder models `p(a | o, z)` — "given o and intent z, produce the chunk."

The observation is fed into both the encoder and the decoder. That's the "C".

---

## 4. How `z` removes the averaging

This is the key idea. Suppose two training demos share the same observation `o`
but took different chunks `a1` (go left) and `a2` (go right).

- The **encoder** sees `(o, a1)` and emits one latent `z1`; it sees `(o, a2)` and
  emits a *different* latent `z2`.
- The **decoder** then learns `(o, z1) -> a1` and `(o, z2) -> a2`. No conflict —
  `z` tells it *which* mode.

So the multimodality is absorbed into `z`. The decoder never has to average,
because the ambiguity is explained away by the latent. At test time we pick one
`z` (the prior mean, `z=0`) and get **one coherent mode**, not a blend.

---

## 5. The three transformer pieces (our `act_model.py`)

```
TRAINING                                    INFERENCE
--------                                    ---------
(o, expert chunk a)                         (o)
       |                                       |
   [CVAE encoder]  -> mu, logvar            z = 0  (skip encoder)
       |  reparam                              |
       z = mu + sigma * eps                    |
       \______________   ________________/    |
                      \ /                      |
                  [main encoder] (o, z -> memory)
                       |
                  [decoder] (K queries cross-attend to memory)
                       |
                  predicted K-action chunk
```

1. **CVAE encoder** (`encode_style`, training only): tokens `[CLS, obs, a_1..a_K]`
   go through a transformer encoder; the `CLS` output is mapped to `(mu, logvar)`.
   This is `q(z | o, a)`.
2. **Main encoder** (`decode`, first half): tokens `[obs, z]` -> "memory".
3. **Decoder** (`decode`, second half): `K` learned query tokens cross-attend to
   memory -> `K` actions. This is `p(a | o, z)`.

At inference we don't have the expert chunk (that's what we're predicting), so we
skip the encoder and set `z = 0`.

---

## 6. The reparameterization trick

We need to sample `z ~ N(mu, sigma)` but also backprop through it. You can't
backprop through a random sample directly. Trick: move the randomness *outside*
the gradient path:

```
z = mu + sigma * eps,   eps ~ N(0,1)
```

`eps` is random but constant w.r.t. the network; gradients flow through `mu` and
`sigma`. In code (`forward`):

```python
std = torch.exp(0.5 * logvar)          # sigma = exp(logvar/2), always positive
z   = mu + std * torch.randn_like(std) # mu + sigma * eps
```

We predict `logvar` (log variance) rather than `sigma` so it's unconstrained
(any real number) and `exp` makes it positive.

---

## 7. The loss and `beta`

```
loss = L1(predicted_chunk, expert_chunk)  +  beta * KL(mu, logvar)
```

- The **L1** term makes the decoder reconstruct the expert's chunk.
- The **KL** term (`kl_divergence`) keeps `z` near `N(0,1)` so that `z=0` is a
  sensible choice at test time.
- **beta** balances them. Too high -> KL dominates -> `z` collapses to the prior
  and carries no information (decoder ignores it = back to averaging). Too low ->
  `z` memorizes / latent space gets ragged and `z=0` is a poor sample.

KL formula (closed form for two Gaussians):

```
KL(N(mu,sigma) || N(0,1)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
```

---

## 8. What happened in OUR runs: KL collapse

In our training logs `train_KL` fell to ~0.000 within ~15 epochs (with beta=10).
That is **posterior collapse**: the encoder gave up using `z`, the decoder learned
to ignore it, and ACT effectively became a *deterministic* chunk predictor.

Why it still worked (86.7%): our state-based task isn't very multimodal once the
target is in the observation — the observation almost fully determines the right
action, so there's little for `z` to disambiguate. The transformer + chunking +
50 Hz did the heavy lifting; the CVAE was along for the ride.

When `z` matters more (vision, richer multimodality, human demos with varied
styles), you'd lower `beta` (~0.1–1) or use KL annealing to keep the latent alive.

---

## 9. Code map (`act_model.py`)

| Concept | Code |
|---|---|
| `q(z\|o,a)` encoder | `encode_style()` (lines 75-85) |
| CLS token -> (mu, logvar) | `self.to_latent`, `.chunk(2)` |
| reparameterization | `forward()` lines 107-108 |
| `z=0` at inference | `forward()` lines 110-111 |
| `p(a\|o,z)` main encoder + decoder | `decode()` (lines 87-97) |
| K action queries | `self.query_embed` + decoder cross-attn |
| KL term | `kl_divergence()` (lines 117-121) |
| L1 + beta*KL | `act_train.py` training loop |

---

## One-line summary

> The CVAE adds a latent "intent" variable `z`: during training the encoder
> assigns different `z` to conflicting-but-valid actions so the decoder never has
> to average them; at test time `z=0` picks one coherent mode. It's the principled
> cure for the multimodality that makes naive behavior cloning mushy.
