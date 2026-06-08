There are **three distinct but tightly linked artifacts** people lump together as "LeCun's world model":

- **LeJEPA** (Balestriero & LeCun, Nov 2025, [arXiv 2511.08544](https://arxiv.org/abs/2511.08544)) — the self-supervised *representation learning* theory + method.
- **LeWorldModel / LeWM** (Maes, Le Lidec, Scieur, LeCun, Balestriero, Mar 2026, [arXiv 2603.19312](https://arxiv.org/abs/2603.19312)) — the actual **world model** that puts LeJEPA's principle into an action-conditioned predictor trained from pixels.
- **"When Does LeJEPA Learn a World Model?"** (May 2026, [arXiv 2605.26379](https://arxiv.org/html/2605.26379v1)) — the identifiability theory proving *when* the recovered latents are the true ones.

These come out of LeCun's research circle around his post-Meta startup, **AMI Labs (Advanced Machine Intelligence)**.

## Why a world model, not an LLM

LeCun's thesis (from his 2022 position paper *"A Path Towards Autonomous Machine Intelligence"*) is that intelligence requires an internal **world model**: a function that predicts how the world evolves, especially as a consequence of actions, so an agent can *plan* by simulating futures rather than reacting token-by-token.

The central design choice is **where you predict**:

- **Generative models** (LLMs, video diffusion, pixel-space world models) predict in *observation space* — the next token or frame in pixels. This wastes capacity on intrinsically unpredictable detail (every leaf, texture, sensor noise).
- **JEPA (Joint Embedding Predictive Architecture)** predicts in a *learned latent space*. Encode observations to abstract embeddings and predict the *embedding* of the future, discarding what is unpredictable.

That single move — predict in latent space, not pixel space — is the whole conceptual backbone.

## JEPA & the collapse problem

Given two views of a signal (a context $x$ and a target $y$ — two crops, or observation-at-$t$ and observation-at-$t{+}1$): an encoder $f_\theta$ maps each to embeddings $s_x = f_\theta(x)$, $s_y = f_\theta(y)$; a predictor $g_\phi$ predicts the target embedding from the context, optionally conditioned on an action $a$: $\hat s_y = g_\phi(s_x, a)$; and an **energy** measures incompatibility: $E(x,y) = \lVert \hat s_y - s_y \rVert^2$.

<div class="figure">
<svg viewBox="0 0 760 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="JEPA / LeWorldModel data flow">
<defs>
<marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L7,3 L0,6 Z" fill="#6E7A86"></path></marker>
<marker id="arrClay" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L7,3 L0,6 Z" fill="#E25A3B"></path></marker>
</defs>
<style>
.box{fill:#FBF4E6;stroke:#E3DFD3;stroke-width:1.5}
.enc{fill:#FBF1EA;stroke:#E4B79E}
.lat{fill:#EAF2F8;stroke:#9FC3E0}
.lbl{font:600 13px 'DM Sans',sans-serif;fill:#21303F}
.sub{font:12px 'DM Sans',sans-serif;fill:#6E7A86}
.flow{stroke:#6E7A86;stroke-width:1.6;fill:none}
.flowC{stroke:#E25A3B;stroke-width:1.8;fill:none;stroke-dasharray:5 4}
.ttl{font:700 12px 'DM Sans',sans-serif;fill:#B85C3E;letter-spacing:.06em}
</style>
<text x="20" y="40" class="sub">observation oₜ</text>
<rect class="box" x="20" y="50" width="70" height="46" rx="10"></rect><text x="34" y="78" class="sub">frame oₜ</text>
<line class="flow" x1="90" y1="73" x2="135" y2="73" marker-end="url(#arr)"></line>
<rect class="box enc" x="135" y="50" width="95" height="46" rx="10"></rect><text x="150" y="78" class="lbl">Encoder f_θ</text>
<line class="flow" x1="230" y1="73" x2="280" y2="73" marker-end="url(#arr)"></line>
<rect class="box lat" x="280" y="50" width="80" height="46" rx="10"></rect><text x="308" y="78" class="lbl">zₜ</text>
<line class="flow" x1="360" y1="73" x2="430" y2="73" marker-end="url(#arr)"></line>
<rect class="box enc" x="430" y="42" width="120" height="62" rx="10"></rect><text x="448" y="68" class="lbl">Predictor g_φ</text><text x="458" y="88" class="sub">(zₜ, aₜ)</text>
<text x="455" y="135" class="sub">action aₜ</text><line class="flow" x1="490" y1="138" x2="490" y2="106" marker-end="url(#arr)"></line>
<line class="flow" x1="550" y1="73" x2="615" y2="73" marker-end="url(#arr)"></line>
<rect class="box lat" x="615" y="50" width="120" height="46" rx="10"></rect><text x="630" y="78" class="lbl">ẑₜ₊₁ (pred.)</text>
<text x="20" y="222" class="sub">observation oₜ₊₁</text>
<rect class="box" x="20" y="232" width="70" height="46" rx="10"></rect><text x="30" y="260" class="sub">frame oₜ₊₁</text>
<line class="flow" x1="90" y1="255" x2="135" y2="255" marker-end="url(#arr)"></line>
<rect class="box enc" x="135" y="232" width="95" height="46" rx="10"></rect><text x="150" y="260" class="lbl">Encoder f_θ</text>
<line class="flow" x1="230" y1="255" x2="615" y2="255" marker-end="url(#arr)"></line>
<rect class="box lat" x="615" y="232" width="120" height="46" rx="10"></rect><text x="628" y="260" class="lbl">zₜ₊₁ (target)</text>
<line class="flowC" x1="675" y1="96" x2="675" y2="232" marker-end="url(#arrClay)" marker-start="url(#arrClay)"></line>
<rect x="540" y="150" width="205" height="38" rx="8" fill="#FBF1EA" stroke="#E4B79E" stroke-width="1.5"></rect>
<text x="556" y="167" class="ttl">PREDICTION ENERGY</text><text x="556" y="182" class="sub">E = ‖ẑₜ₊₁ − zₜ₊₁‖²</text>
<rect x="280" y="150" width="230" height="46" rx="8" fill="#EAF2F8" stroke="#11A79C" stroke-width="1.5" stroke-dasharray="4 3"></rect>
<text x="296" y="170" class="ttl">SIGReg (anti-collapse)</text><text x="296" y="187" class="sub">embeddings → isotropic 𝒩(0, I)</text>
<line class="flowC" x1="320" y1="96" x2="335" y2="150"></line><line class="flowC" x1="660" y1="232" x2="470" y2="196"></line>
</svg>
<p class="cap">JEPA / LeWorldModel data flow. The predictor models latent dynamics conditioned on the action; the prediction energy is computed entirely in embedding space, while <b>SIGReg</b> pulls the embedding distribution toward an isotropic Gaussian to prevent collapse.</p>
</div>

This is an **energy-based model**: low energy = "these go together," and in the world-model case the predictor *is* the dynamics model. The fatal failure mode is **collapse**: if $f_\theta$ maps everything to a constant, prediction loss is trivially 0 and the energy landscape is flat. Every JEPA's core engineering problem is preventing collapse while keeping a simple prediction loss.

| Method | Anti-collapse mechanism |
|---|---|
| SimCLR / contrastive | Negative samples push apart (needs huge batches) |
| BYOL / SimSiam | Stop-gradient + EMA "teacher" + predictor asymmetry |
| VICReg / Barlow Twins | Explicit variance + covariance penalties |
| I-JEPA (2023, images) | Masked prediction + EMA target encoder |
| V-JEPA / V-JEPA 2 (2024–25) | Masked spatiotemporal prediction, scaled to video |
| DINO-WM (2024) | Frozen pre-trained DINO features as the latent space |

All of these are *heuristic*. LeCun's 2025–26 line removes the heuristics and replaces them with a single principled regularizer.

## LeJEPA: making it provable

Key theoretical claim: **the isotropic Gaussian is the unique embedding distribution that minimizes worst-case downstream prediction risk.** If you don't know the downstream task, the optimal thing your embeddings can do is be distributed as $\mathcal N(0, I)$ — maximally spread, no privileged directions, no collapse.

So instead of *ad hoc* variance/covariance terms, you directly regularize toward isotropic Gaussian via **SIGReg (Sketched Isotropic Gaussian Regularization)**: draw random unit directions and project embeddings onto each (a distribution is isotropic Gaussian iff every 1-D projection is standard Gaussian — Cramér–Wold); then test each projection against $\mathcal N(0,1)$ with a characteristic-function goodness-of-fit statistic (Epps–Pulley):

<div class="eq">\[ \text{SIGReg} = \mathbb{E}_{u\sim \mathbb S^{D-1}} \int \big|\, \hat\varphi_u(t) - e^{-t^2/2} \,\big|^2 \, w(t)\,dt \]</div>

This is **linear $O(N)$** in time/memory — no negatives, no teacher, no stop-gradient, no EMA. The total objective has essentially **one** hyperparameter:

<div class="eq">\[ \mathcal L_{\text{LeJEPA}} = \underbrace{\lVert g_\phi(f_\theta(x)) - \mathrm{sg}[f_\theta(y)]\rVert^2}_{\text{prediction}} + \lambda\, \underbrace{\text{SIGReg}\big(\{f_\theta(\cdot)\}\big)}_{\text{isotropic-Gaussian regularizer}} \]</div>

A faithful PyTorch reconstruction of SIGReg (illustrative, not copied from the official repo):

```python
import torch

def epps_pulley_1d(p, ts, weight):
    # p: (N,) standardized projections; compare ECF to N(0,1) CF
    tp = ts[:, None] * p[None, :]                 # (T, N)
    re = torch.cos(tp).mean(dim=1)                # Re phi_hat(t)
    im = torch.sin(tp).mean(dim=1)                # Im phi_hat(t)
    target = torch.exp(-0.5 * ts**2)              # CF of N(0,1)
    diff2 = (re - target)**2 + im**2              # |phi_hat - phi_gauss|^2
    return (diff2 * weight).sum()

def sigreg(Z, n_dirs=256, n_freq=64, t_max=5.0):
    N, D = Z.shape
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)       # center / scale per dim
    U = torch.randn(D, n_dirs, device=Z.device)
    U = U / U.norm(dim=0, keepdim=True)           # random unit directions
    P = Z @ U                                      # sketched 1-D projections
    P = (P - P.mean(0)) / (P.std(0) + 1e-6)
    ts = torch.linspace(0.05, t_max, n_freq, device=Z.device)
    w = torch.exp(-0.5 * ts**2)                    # Gaussian frequency weighting
    return torch.stack([epps_pulley_1d(P[:, k], ts, w) for k in range(n_dirs)]).mean()
```

## LeWorldModel — the actual world model

Authors: **Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero** (Mila / NYU / Samsung SAIL / Brown). It is the **first JEPA world model that trains stably end-to-end from raw pixels** with only **two loss terms** — applying the LeJEPA insight to action-conditioned dynamics.

- **Encoder** $z_t = \mathrm{enc}(o_t)$ — maps a pixel frame to a **single ~192-dim token** (≈200× fewer than DINO-WM's patch tokens).
- **Predictor** $\hat z_{t+1} = \mathrm{pred}(z_t, a_t)$ — latent dynamics conditioned on action $a_t$.
- **~15M parameters**, trainable on a **single GPU in a few hours**.

<div class="eq">\[ \mathcal L_{\text{LeWM}} = \underbrace{\lVert \mathrm{pred}(z_t, a_t) - z_{t+1}\rVert^2}_{\text{next-embedding prediction}} + \lambda\, \underbrace{\text{SIGReg}(Z)}_{\text{Gaussian latent}} \]</div>

This drops the hyperparameter count **from six to one** vs. the only prior end-to-end JEPA world model. Empirically it plans **up to 48× faster** than foundation-model-based world models, stays competitive on diverse 2D/3D control, encodes real physical structure (probeable), and reliably spikes its prediction error on physically implausible events ("surprise").

What makes it a *world model* rather than just an encoder: roll the predictor forward over candidate action sequences and pick the one whose imagined latent trajectory reaches the goal (CEM-style MPC, entirely in latent space):

```python
@torch.no_grad()
def plan_mpc(enc, pred, o_now, z_goal, action_dim, horizon=10,
             n_samples=512, iters=4, elite=64):
    z = enc(o_now)                                       # (1, d)
    mu  = torch.zeros(horizon, action_dim)
    std = torch.ones(horizon, action_dim)
    for _ in range(iters):
        A = mu + std * torch.randn(n_samples, horizon, action_dim)
        zt = z.expand(n_samples, -1).clone()
        cost = torch.zeros(n_samples)
        for h in range(horizon):
            zt = pred(zt, A[:, h])                       # imagine next latent
            cost += ((zt - z_goal)**2).sum(-1)           # distance-to-goal in latent
        idx = cost.topk(elite, largest=False).indices    # keep best action seqs
        mu, std = A[idx].mean(0), A[idx].std(0) + 1e-4    # refit sampling dist
    return mu[0]                                          # execute first action, replan
```

Same family as **PlaNet / Dreamer** MPC-over-imagination, but the dynamics live in a non-generative, SIGReg-regularized latent rather than a reconstructive RSSM.

## When does it learn a world model?

The theory paper (arXiv 2605.26379) answers *when the recovered latents are the true hidden variables* (identifiability):

- **Theorem 5.1 (Linear Identifiability).** Under a Gaussian world model, LeJEPA recovers the true latents **up to an orthogonal rotation** $Q$; the alignment loss obeys $\mathcal L(h) \ge 2(1-\rho)n$, with equality iff $h(z) = Qz$.
- **Theorem 5.2 (Gaussian Uniqueness).** The Gaussian is the *unique* latent distribution (in the class) giving linear identifiability — the converse that justifies SIGReg's target.
- **Theorem 5.3 (Graceful degradation).** Approximate recovery with error $\sim \delta / (2\rho(1-\rho))$ + regularization error.

**Assumptions:** independent latent components/transitions, stationarity, and additive Ornstein–Uhlenbeck noise $z' = \rho z + \sqrt{1-\rho^2}\,\eta$. For Gaussian latents the transition operator's eigenfunctions are **Hermite polynomials** with eigenvalues $\rho^d$, so any nonlinear map strictly reduces cross-correlation.

> **Honest limitation:** the guarantees hold under Gaussian latents + OU dynamics + good exploration, and only for the **encoder** — not the action-conditioned predictor. So "LeWM provably learns the true world model" is **not** yet established; the dynamics side remains empirical.

## Relation to other work

Direct ancestry (LeCun's own line): I-JEPA → V-JEPA / V-JEPA 2 → LeJEPA → LeWorldModel — "remove every heuristic."

| Work | Relation |
|---|---|
| Ha & Schmidhuber, "World Models" (2018) | VAE + MDN-RNN dynamics + controller. The original; *generative* latent. |
| PlaNet / Dreamer v1–v3 | RSSM latent dynamics with a *reconstruction* loss. LeWM shares latent-MPC but drops reconstruction. |
| DINO-WM (2024) | Closest baseline; uses *frozen* DINO features. LeWM trains the encoder end-to-end with a ~200× smaller token. |
| Genie / Genie 2, Sora-as-world-model | *Generative, pixel-space* — exactly what LeCun argues against. |
| Causal-JEPA, Var-JEPA, Rectified Lp-JEPA (2026) | Community spinoffs extending JEPA + distributional regularizer. |

The unifying contrast: **predict-in-latent + distributional regularizer (LeCun camp)** vs. **predict-in-observation-space / generative (LLM + video-diffusion camp)**.

## The AMI startup

**AMI Labs (Advanced Machine Intelligence)**, founded by LeCun after leaving Meta, raised a **~$1.03B seed round** (announced ~March 2026) — among the largest AI seed rounds ever, on an explicit thesis: build **world models**, a contrarian bet *against* LLMs as the path to human-level intelligence. Critical coverage stresses that the funding-round framing oversells novelty — the theory only guarantees encoder-side recovery under idealized assumptions, and the world-model dynamics side remains empirical.

## Sources

- [LeWorldModel — arXiv 2603.19312](https://arxiv.org/abs/2603.19312) · [project page](https://le-wm.github.io/)
- [LeJEPA — arXiv 2511.08544](https://arxiv.org/abs/2511.08544) · [code (rbalestr-lab/lejepa)](https://github.com/rbalestr-lab/lejepa)
- [When Does LeJEPA Learn a World Model? — arXiv 2605.26379](https://arxiv.org/html/2605.26379v1)
- [I-JEPA (Meta AI blog)](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/) · [AMI Labs — TechCrunch](https://techcrunch.com/2026/01/23/whos-behind-ami-labs-yann-lecuns-world-model-startup/)
