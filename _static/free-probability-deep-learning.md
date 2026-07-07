<!-- DRAFT — Post 1 of the "Random Matrices Meet Deep Learning" series
     (roadmap: _static/rmt-deep-learning-series.md). Not yet registered in
     build.py / MD_POSTS. To publish: add an MD_POSTS entry and run python3
     build.py. Slug to rename to "rmt-free-probability" on publish. -->

> **Status: draft / outline — Post 1 of the *Random Matrices Meet Deep Learning*
> series.** This page is the theory backbone for the series: a thorough table of
> contents, a section-by-section plan, and a curated resource list (theory + code).
> Prose, figures and worked examples get filled in per section before publishing.
> The unexpected hooks (double descent as a Marchenko–Pastur edge, heavy tails that
> predict test accuracy, scaling laws from a power-law spectrum, …) get their own
> later posts — see the roadmap.

Free probability is the strange and beautiful corner of mathematics that asks:
*what is "independence" when your random variables don't commute* — when they are
large matrices instead of scalars? Dan-Virgil Voiculescu's answer, **freeness**,
turns out to be exactly the notion that governs the eigenvalue spectra of large
random matrices. And large random matrices are everywhere in deep learning:
weight matrices, Jacobians, Hessians, kernels. This post is a guided tour from
the abstract algebra to concrete predictions about the loss-surface geometry and
trainability of deep networks.

---

## Table of contents

1. **Why free probability for deep learning?** — the one-paragraph pitch
2. **From scalars to operators** — classical vs non-commutative probability
   1. Random variables as operators; the trace as expectation
   2. Spectral distribution of a (random) matrix
   3. Why ordinary independence is the wrong tool for matrices
3. **Freeness: the central idea**
   1. Voiculescu's definition of free independence
   2. Intuition: "free" = independent + maximally non-aligned eigenbases
   3. Asymptotic freeness of random matrices (Voiculescu's theorem)
   4. Worked micro-example: free vs classical for $2\times2$ blocks
4. **Free products: where free random variables live**
   1. Classical independence ↔ tensor product; free independence ↔ **free product**
   2. The free product of non-commutative probability spaces $(\mathcal{A}, \tau) = \ast_i (\mathcal{A}_i, \tau_i)$
   3. Voiculescu's motivation: free group factors $L(\mathbb{F}_n)$ and the isomorphism problem
   4. From free product to free convolution: the laws of $a+b$ and $ab$ for free $a,b$
5. **The analytic machinery: transforms**
   1. Cauchy / Stieltjes transform $G(z)$ and inversion (fix the $z-t$ sign convention)
   2. Moments, free cumulants and non-crossing partitions (the combinatorial backbone)
   3. The **R-transform** — generating function of free cumulants; linearizes free *additive* convolution
   4. The **S-transform** — linearizes free *multiplicative* convolution (needs nonzero mean)
6. **Free convolution**
   1. Free additive convolution $\mu \boxplus \nu$ via the R-transform
   2. Free multiplicative convolution $\mu \boxtimes \nu$ via the S-transform
   3. Subordination functions and why convolutions are hard to compute in practice
   4. Numerically computing a free convolution (fixed-point / subordination)
7. **Limiting spectral distributions**
   1. The **free Central Limit Theorem** → Wigner semicircle law
   2. The **free Poisson** → Marchenko–Pastur law (Wishart matrices)
   3. Free stability, free infinite divisibility (the "free" analogues of Gaussian/Poisson/stable)
   4. Sums and products: what $\boxplus$ and $\boxtimes$ do to the support
8. **Bridge I — the Hessian as a sum: $\boxplus$ and the R-transform**
   1. The classical decomposition $H = H_0 + H_1$ (Gauss–Newton/Fisher term + residual term)
   2. Wishart $\boxplus$ Wigner: Pennington & Bahri's loss-surface model
   3. Energy vs index: how the spectrum (and negative eigenvalues) move with loss
   4. What it predicts about saddle points and critical-point structure
9. **Bridge II — the Jacobian as a product: $\boxtimes$ and the S-transform**
   1. Deep network input–output Jacobian as a product of layer matrices
   2. S-transform of a product = product of S-transforms → whole singular spectrum
   3. **Dynamical isometry**: concentrating all singular values near 1
   4. Choice of nonlinearity, weight init (orthogonal vs Gaussian), depth scaling
   5. Spectral universality: limiting distributions that survive depth $\to \infty$
10. **Bridge III — kernels, nonlinear RMT and the NTK**
    1. Single nonlinear Gram matrix: a deformed-MP self-consistent equation (Pennington–Worah)
    2. Iterating it layer by layer: Conjugate Kernel & NTK spectra (Fan–Wang)
    3. Spiked models / outliers: the BBP transition (top eigenvalues, "spikes")
    4. Gaussian equivalence and where it holds / breaks
11. **Empirical playground (code)**
    1. Simulating asymptotic freeness: two random matrices, free vs classical sum
    2. Numerically evaluating $\boxplus$ / $\boxtimes$ and overlaying on histograms
    3. Measuring a real network's Hessian spectrum with **PyHessian** (SLQ)
    4. Jacobian singular values vs the S-transform prediction across depth
    5. Watching weight-matrix spectra drift from Marchenko–Pastur during training
12. **Tensions and open questions**
    1. Why empirical Hessians *don't* look like clean Wishart/MP (Gaussian-equivalence limits)
    2. Heavy-tailed self-regularization (HT-SR) vs the bulk RMT picture
    3. Finite width, correlated weights, structure after training
13. **Resources** — theory, surveys, papers, code

---

## Section-by-section plan

### 1. Why free probability for deep learning?
One tight motivating paragraph + a teaser figure. The hook: three central objects
of DL theory are large random matrices whose spectra we care about — the **Hessian**
(loss geometry / optimization), the **input–output Jacobian** (signal propagation /
trainability), and **kernels** (CK/NTK, generalization). Free probability is the
algebra of *combining* such matrices: sums (Hessian decomposition) via $\boxplus$,
products (deep Jacobian) via $\boxtimes$. Promise: by the end, the reader can read a
spectrum and predict how depth/init/nonlinearity reshape it.

### 2. From scalars to operators
Keep it gentle. Define a non-commutative probability space $(\mathcal{A}, \tau)$
with $\tau$ a trace playing the role of $\mathbb{E}$. For an $N\times N$ Hermitian
matrix $M$, the normalized trace $\frac1N \mathrm{Tr}$ recovers the empirical
spectral distribution $\mu_M = \frac1N \sum_i \delta_{\lambda_i}$. Punchline of the
section: *adding two matrices mixes their eigenvectors*, so the spectrum of $A+B$ is
**not** a classical convolution of the two spectra — we need a new notion of
independence.

### 3. Freeness: the central idea
State Voiculescu's definition (alternating centered products of elements from free
subalgebras have trace zero). Then immediately translate to intuition: freeness is
what you get when two matrices are in **generic relative position** — random
rotation between their eigenbases. State the **asymptotic freeness** theorem:
independent random matrices (e.g. one fixed, one Haar-rotated, or two independent
Wigner/Wishart) become asymptotically free as $N\to\infty$. This is the theorem that
licenses everything downstream. Small concrete example to make "free $\ne$
independent" tangible.

### 4. Free products: where free random variables live
The construction that *realizes* freeness. Classical probability models
independence with the **tensor product** of spaces; free probability models
freeness with the **free product**. Given non-commutative probability spaces
$(\mathcal{A}_i, \tau_i)$, their free product $(\mathcal{A}, \tau) = \ast_i
(\mathcal{A}_i, \tau_i)$ is the algebra they generate with the free relation baked
in — the canonical copies of the $\mathcal{A}_i$ sitting inside are automatically
free. So "freeness" is not just an abstract rule on traces (§3); the free product
is the concrete *home* where free random variables actually live (realized on the
**full Fock space** via creation/annihilation operators — the free analogue of the
Gaussian/Bosonic Fock space).

Two threads to pull, one historical and one that we need downstream:
- **Historical / motivational.** Voiculescu introduced free probability precisely
  to attack the **free group factor isomorphism problem**: are the von Neumann
  algebras $L(\mathbb{F}_m)$ and $L(\mathbb{F}_n)$ isomorphic for $m \neq n$? The
  group von Neumann algebra of a free product of groups is the (von Neumann) free
  product of their algebras, and freeness is exactly the structure the free
  generators exhibit. (The problem is still open — a great "this math is alive"
  aside.)
- **Operational.** The free product is *where free convolution happens*: $\mu
  \boxplus \nu$ and $\mu \boxtimes \nu$ (§6) are defined as the distributions of
  $a+b$ and $ab$ when $a,b$ are the canonical free copies of variables with laws
  $\mu,\nu$ living in the free product. (Under the trace, $ab$ has the same
  spectral law as the self-adjoint $\sqrt{a}\,b\,\sqrt{a}$ that §6 uses — same
  $\boxtimes$, just a symmetrized representative.) And **asymptotic freeness**
  (§3) is the
  statement that large random matrices *approximate* elements of a free product —
  which is the whole reason any of this touches deep learning. Also flag free
  products of measures / graphs (free random walks) as the combinatorial cousin.

Keep it short and intuitive — one schematic (tensor product vs free product), no
operator-algebra prerequisites; the payoff is that the reader sees free
convolution as "compute in the free product, then read off the law."

### 5. The analytic machinery: transforms
The toolbox. Define $G_\mu(z) = \int \frac{d\mu(t)}{z-t}$ (pin the $z-t$ convention
once — some texts use $t-z$). Introduce **free cumulants** and **non-crossing
partitions** *first*, since the R-transform is exactly their generating function —
that is *why* it linearizes $\boxplus$ (the free analogue of the moment–cumulant
relations). Then the **R-transform**: let $K = G^{-1}$ be the *compositional* inverse
of the Cauchy transform (so $G(K(z)) = z$ — **not** the reciprocal $1/G$); then
$R(z) = K(z) - \frac1z$, and $R_{\mu\boxplus\nu} = R_\mu + R_\nu$. The **S-transform**
is the multiplicative analogue (defined only when $\int t\,d\mu \neq 0$).
Box: "R-transform : freeness :: log-characteristic function : classical
independence."

### 6. Free convolution
Define $\mu \boxplus \nu$ (spectrum of $A+B$ for free $A,B$) via $R_{\mu\boxplus\nu} =
R_\mu + R_\nu$, and $\mu \boxtimes \nu$ (spectrum of $\sqrt{A}\,B\,\sqrt{A}$) via
$S_{\mu\boxtimes\nu} = S_\mu \cdot S_\nu$. Be honest that closed forms are rare:
inverting $G$ is hard, so practitioners use **subordination functions** and
fixed-point iteration. Set up the code in §11.2.

### 7. Limiting spectral distributions
The "named distributions." Free CLT: the *rescaled* sum
$(a_1 + \dots + a_n)/\sqrt{n}$ of free, centered, variance-1 laws $\to$
**semicircle** (the free Gaussian) — the $1/\sqrt n$ is essential, without it the
support blows up. Free Poisson limit $\to$ **Marchenko–Pastur** (Wishart /
sample-covariance spectra), parameterized by aspect ratio. (Free stability / free
infinite divisibility get a one-line remark only — lovely, but no payoff in the DL
bridges below.) Tie each to a DL object: semicircle ↔ symmetric weight/Wigner terms,
MP ↔ Gram/covariance/Gauss–Newton terms.

### 8. Bridge I — the Hessian as a sum
The first real payoff. Use the **classical** Hessian decomposition $H = H_0 + H_1$
(Schraudolph; LeCun et al.): $H_0$ = the positive-semidefinite Gauss–Newton/Fisher
("functional", Wishart-like) term, $H_1$ = the residual-weighted term (indefinite,
Wigner-like). The decomposition is *not* new; **Pennington & Bahri's** contribution
is to *model these two pieces as freely independent* and compute the spectrum of the
sum via $\boxplus$ — a stylized **Wishart $\boxplus$ Wigner** density (a model, not a
derivation from the true Hessian). Key qualitative predictions:
the spectrum interpolates between MP-like (low loss, an eigenvalue gap, mostly
convex) and semicircle-like (high loss, negative eigenvalues / saddles emerge), with
the index of critical points rising with energy. Reproduce their energy–index curve.

### 9. Bridge II — the Jacobian as a product
The second payoff, and the cleanest use of the S-transform. The input–output
Jacobian of an $L$-layer net is a **product** $J = \prod_l D_l W_l$. The S-transform
machinery acts on the limiting distribution of the **squared singular values** of
$J$ — i.e. on the spectrum of $JJ^{\mathsf T}$, built from the PSD factors
$(D_l W_l)(D_l W_l)^{\mathsf T}$ (not the signed factors themselves). The load-bearing
hypothesis is that these per-layer PSD factors are **asymptotically free** of one
another (from independence of the $W_l$ + Haar/Gaussian structure, with $D_l$ free
of $W_l W_l^{\mathsf T}$); *given* that, $\boxtimes$-multiplicativity of $S$ gives the
whole spectrum without ever multiplying matrices. This yields **dynamical isometry**:
tune nonlinearity + (orthogonal)
weight init so the entire spectrum concentrates near 1, killing exploding/vanishing
signals and letting *very* deep nets train. Cover the sigmoid-resurrection result
and the depth-independent **universal** limiting spectra.

### 10. Bridge III — kernels, nonlinear RMT and the NTK
Round out the picture — but scope it honestly, this is harder than Bridges I+II
combined, so aim for "the spectra exist and are computable, here is the shape," not
full derivations. Two *distinct* results, kept separate:
- **Pennington & Worah (NeurIPS 2017)** — the spectrum of a *single* nonlinear Gram
  matrix $\frac1m f(WX)^{\mathsf T} f(WX)$ is a **deformed Marchenko–Pastur** law,
  the solution of a quartic self-consistent (moment) equation governed by two
  parameters $(\eta,\zeta)$ of the nonlinearity. ("MP map" is a slogan; the object is
  a self-consistent equation, not literally MP.)
- **Fan & Wang (NeurIPS 2020)** — *iterate* that map layer by layer (a multiplicative
  free convolution with an MP-like law) to get the **Conjugate Kernel** spectrum, and
  extend it via fixed-point equations to the **NTK** in the linear-width regime.

Briefly add **spiked models / the BBP transition** — a finite-rank perturbation of a
bulk pops out outlier eigenvalues past a threshold; this is the tool behind the
"spikes" seen in real Hessian/weight spectra (§11.5). Close with **Gaussian
equivalence** as the bridge that makes nonlinear features tractable, and where it
breaks.

### 11. Empirical playground (code)
Make it runnable. Planned snippets (numpy / scipy / pytorch):
- **Asymptotic freeness demo**: sample two large symmetric random matrices, compare
  the histogram of $A+B$ against the *classical* convolution of the marginals and
  against the *free* convolution — show only the free one matches.
- **Numerical free convolution**: implement $\boxplus$ via Cauchy-transform
  subordination fixed point; overlay semicircle $\boxplus$ semicircle, MP $\boxplus$
  semicircle.
- **Real Hessian spectrum**: use **PyHessian** (Stochastic Lanczos Quadrature) on a
  small CNN; ask "does the bulk match a Wishart $\boxplus$ Wigner fit?" — frame it as
  *partial agreement* (sets up the §12 tension, don't oversell the match).
- **Jacobian vs S-transform**: build deep nets at varying depth/init, plot empirical
  singular-value histogram against the S-transform prediction; show
  orthogonal+tuned-gain isometry vs Gaussian spreading.
- **Training drift**: track a weight matrix's spectrum from Marchenko–Pastur at init
  toward heavier-tailed/spiked shapes during training.

### 12. Tensions and open questions
Don't oversell. Be explicit that empirical Hessians often *don't* show clean MP/
Wishart bulks; discuss the Gaussian-equivalence caveats, **heavy-tailed
self-regularization** (Martin & Mahoney) as an alternative lens, and the gap between
$N\to\infty$ free-probability idealizations and finite-width, correlated, post-training
reality.

---

## Resources

### Foundations of free probability (theory)
- Voiculescu, Dykema & Nica — *Free Random Variables* (1992): the founding monograph
  (R-transform, free convolution, asymptotic freeness).
- Nica & Speicher — *Lectures on the Combinatorics of Free Probability* (2006): the
  standard graduate text; free cumulants & non-crossing partitions.
- Mingo & Speicher — *Free Probability and Random Matrices* (2017): the RMT-facing
  text, ideal for this post's angle.
- Tao, *Topics in Random Matrix Theory* — free-probability chapter (readable intro):
  <https://terrytao.wordpress.com/tag/free-probability/>
- Voiculescu, *Limit laws for random matrices and free products* (1991) — the
  asymptotic-freeness theorem that licenses Haar-conjugation → freeness (cited by
  §3, §4, §9).
- **Free products & free group factors** (§4): Voiculescu, Dykema & Nica cover the
  free-product construction; see also Voiculescu's *Symmetries of some reduced free
  product C\*-algebras* (1985) for the origin, and any survey of the still-open
  $L(\mathbb{F}_n)$ isomorphism problem.
- **Subordination** (the analytic backbone of numerical free convolution): Biane,
  *Processes with free increments* (1998); Belinschi–Bercovici; Chistyakov–Götze.
- Survey on free convolution & subordination (numerical difficulty / subordination
  functions): r-Free Convolution and Variance Function,
  <https://www.mdpi.com/2075-1680/14/2/128>

### Free probability ↔ deep learning (core papers)
- Pennington & Bahri, *Geometry of Neural Network Loss Surfaces via Random Matrix
  Theory*, ICML 2017 — Hessian = Wishart $\boxplus$ Wigner.
  <https://proceedings.mlr.press/v70/pennington17a/pennington17a.pdf>
- Pennington, Schoenholz & Ganguli, *Resurrecting the Sigmoid in Deep Learning
  through Dynamical Isometry: Theory and Practice*, NeurIPS 2017 — S-transform for
  the Jacobian spectrum. <https://arxiv.org/abs/1711.04735>
- Pennington, Schoenholz & Ganguli, *The Emergence of Spectral Universality in Deep
  Networks*, AISTATS 2018. <https://arxiv.org/abs/1802.09979>
- Pennington & Worah, *Nonlinear Random Matrix Theory for Deep Learning*, NeurIPS
  2017 — deformed-MP spectrum of a single nonlinear Gram matrix (journal-extended
  version: <https://arxiv.org/abs/1904.13063>).
- Fan & Wang, *Spectra of the Conjugate Kernel and Neural Tangent Kernel for
  linear-width neural networks*, NeurIPS 2020. <https://arxiv.org/abs/2005.11879>
- Xiao et al., *Dynamical Isometry and a Mean Field Theory of CNNs: How to Train
  10,000-Layer Vanilla CNNs*, ICML 2018. (depth via isometry, applied)
- Tarnowski et al., *Dynamical Isometry is Achieved in Residual Networks in a
  Universal Way for any Activation Function*. <https://arxiv.org/abs/1809.08848>
- Pastur, *Appearance of Random Matrix Theory in Deep Learning* (2020 survey).
  <https://arxiv.org/abs/2102.06740>
- Murray, Granziol et al., *Universal characteristics of deep neural network loss
  surfaces from random matrix theory*. <https://arxiv.org/abs/2205.08601>

### Loss-surface precursors (the energy–index / saddle picture of Bridge I)
- Dauphin et al., *Identifying and attacking the saddle point problem in
  high-dimensional non-convex optimization*, NeurIPS 2014.
- Choromanska et al., *The Loss Surfaces of Multilayer Networks*, AISTATS 2015 —
  the spin-glass model that the energy–index story descends from.
- Baskerville, Keating et al. — universal spin-glass/RMT Hessian companion to
  Murray–Granziol above.

### Alternative / complementary lens (heavy tails)
- Martin & Mahoney, *Implicit Self-Regularization in Deep Neural Networks: Evidence
  from Random Matrix Theory* (Heavy-Tailed Self-Regularization).
- `weightwatcher` — open-source tool for HT-SR weight-spectrum diagnostics:
  <https://github.com/CalculatedContent/WeightWatcher>

### Empirical / code
- **PyHessian** — Hessian eigenvalues, trace, and full spectral density (SLQ) for
  PyTorch nets. Paper: <https://arxiv.org/abs/1912.07145> · Code:
  <https://github.com/amirgholami/PyHessian>
- Asymptotic-freeness & free-convolution numerics: implement Cauchy-transform
  subordination in numpy/scipy (planned in §11; see Mingo–Speicher Ch. on
  computation and the subordination survey above).
- `RMT4DL` / random-matrix demos and the loss-surface reproductions accompanying the
  Pennington papers (search the authors' code releases).

---

### Notes / TODO before publishing
- Decide depth/length: this could be one long post or a 2-part series (theory →
  empirical).
- Draw 2–3 inline SVG figures in the blog's watercolor style: (a) eigenbasis
  mixing under $A+B$, (b) energy–index curve (Hessian), (c) S-transform/Jacobian
  isometry across depth.
- Pick 1 hero code demo to actually run and screenshot (likely the
  asymptotic-freeness histogram — cheap and visually convincing).
- Categories on publish: `["Research"]` (maybe add a future "Math / Theory" section).
- Tags: `["Free Probability", "Random Matrix Theory", "Deep Learning Theory",
  "Hessian"]`.
