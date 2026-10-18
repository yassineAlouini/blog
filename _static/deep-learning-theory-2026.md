<p class="eyebrow">Research notes · 18 October 2026</p>

What happened to deep learning theory after *The Principles of Deep Learning
Theory*? The most interesting development is our growing ability to describe
how a network's representations change during training, and to connect those
changes to measurable outcomes: training speed, scaling with compute, and the
usefulness of depth.

Roberts, Yaida and Hanin's book offers a particular starting point. Treat a neural
network as a system with many interacting degrees of freedom, identify a
tractable limit, then calculate the corrections that make the system interesting.
That statistical-physics perspective remains productive. This article follows
six developments closest to it, rather than attempting a catalogue of every
branch of learning theory.

The dates below distinguish publication venues from the first appearance of
preprints. A 2026 publication may develop work first circulated earlier. Each
result also comes with a scope: which architecture, training procedure or limit
makes its conclusions possible.

<nav aria-label="Article contents">
<ol>
<li><a href="#starting-point">The book's starting point</a></li>
<li><a href="#feature-dynamics">Following feature learning through time</a></li>
<li><a href="#depth-data-width">Depth, data and width together</a></li>
<li><a href="#parameterization">Scaling rules that guide training</a></li>
<li><a href="#scaling-laws">Explaining scaling-law exponents</a></li>
<li><a href="#optimization">Optimization at the edge of stability</a></li>
<li><a href="#in-context">When depth helps in-context learning</a></li>
<li><a href="#reading">What to read next</a></li>
</ol>
</nav>

<h2 id="starting-point">The book's starting point</h2>

In the book's setting, infinitely wide networks supply a Gaussian starting
description. Finite-width corrections introduce interactions, while depth can
amplify their accumulated effect. The depth-to-width ratio controls departures
from the Gaussian description. This leads to a theory of representation
learning, signal propagation and architectural inductive bias.
[The Principles of Deep Learning Theory, Cambridge University Press, 2022](https://arxiv.org/abs/2106.10165).

One distinction matters for everything that follows. A **Gaussian process**
describes a distribution over functions. A **neural tangent kernel**, or NTK,
describes how parameter changes affect predictions. They are related objects in
wide-network theory, but they answer different questions.

For a scalar-output network, the empirical NTK is

$$
K_t(x,x') = \nabla_\theta f_{\theta_t}(x)^\top
             \nabla_\theta f_{\theta_t}(x').
$$

With the squared loss $\mathcal{L}=\frac12\sum_j(f(x_j)-y_j)^2$ and gradient flow,
the training predictions satisfy

$$
\frac{d f_t(x_i)}{dt}
= -\sum_j K_t(x_i,x_j)\bigl(f_t(x_j)-y_j\bigr).
$$

This identity follows directly from the chain rule. Approximating $K_t$ by a
fixed $K_0$ makes the dynamics much easier to solve. Understanding when that
approximation fails, and how the kernel then evolves, is a central question in
the next generation of theories.

<h2 id="feature-dynamics">1. Following feature learning through time</h2>

**Dynamical mean-field theory (DMFT)** replaces the description of every parameter
with equations for collective quantities, including correlations of activations
and gradients across examples and training times. These quantities determine
the evolution of representations and predictions together.

Bordelon and Pehlevan derived self-consistent equations for infinite-width
networks that retain feature learning. Their analysis includes regimes where
the static NTK approximation and leading-order perturbation theory fail while
the self-consistent description remains accurate.
[Foundational work, first posted in 2022](https://arxiv.org/abs/2205.09653).

This is a useful correction to the intuition that infinite width necessarily
eliminates feature learning: the outcome depends on how the network and its
updates are scaled as width grows.

The approach has also reached transformers. Bordelon, Chaudhry and Pehlevan
analyze training limits obtained by increasing attention-head dimension, head
count or depth. These limits can have different statistical descriptions and
different learned features. Their model allows attention to change throughout
training, and they test convergence to the predicted limits numerically.
[Infinite Limits of Multi-head Transformer Dynamics, NeurIPS 2024](https://arxiv.org/abs/2405.15712).

**What advances beyond the book:** collective descriptions can follow substantial
representation changes. The limits and parameterizations remain part of the
claim; the equations are not an exact simulator of any finite transformer.

<h2 id="depth-data-width">2. Depth, data and width together</h2>

A particularly direct continuation is Hanin and Zlokapa's **Gibbs Measures from
Deep Shaped Multilayer Perceptrons**, published in *Physical Review Letters* in
February 2026. It studies Bayesian posteriors in networks with specially scaled
nonlinearities using a diagrammatic, perturbative calculation.

With depth $L$, sample count $P$ and width $N$, a controlling quantity is

$$
\rho = \frac{LP}{N}.
$$

When $\rho$ tends to zero, the posteriors coincide with those of a kernel method.
When it approaches a positive value, the analysis produces data-dependent
deformations of the kernels and explicit first-order feature corrections. The
large-depth, large-width and large-data limits do not generally commute.
[Hanin and Zlokapa, PRL 136, 067301](https://doi.org/10.1103/xm8d-v66z).

**What advances beyond the book:** sample count joins architecture in determining
the learning regime. This ratio is a result for the paper's shaped-network
Bayesian setting, not a universal feature-learning threshold for SGD or Adam.

<h2 id="parameterization">3. Scaling rules that guide training</h2>

The choice of a large-network limit has practical consequences. **Maximal-update
parameterization**, usually written $\mu$P, scales initialization and training
hyperparameters with width so that feature updates remain meaningful. It also
motivates tuning hyperparameters on smaller networks and transferring them to
larger ones. The underlying infinite-width feature-learning construction dates
to 2021.
[Yang and Hu's research explanation](https://www.microsoft.com/en-us/research/blog/on-infinitely-wide-neural-networks-that-exhibit-feature-learning/).

Recent work asks how to preserve useful learning when **depth also changes**.
In **CompleteP**, Dey and colleagues study a parameterization that combines
depth-wise hyperparameter transfer with non-lazy learning in all layers: feature
changes remain meaningfully nonlinear instead of staying close to a
linearization. Their transformer experiments report 12–34% improvements in
compute efficiency over the compared baselines.
[Don't be lazy: CompleteP enables compute-efficient deep transformers, NeurIPS 2025](https://arxiv.org/abs/2505.01618).

**Why this matters:** the theory suggests concrete training rules. The reported
efficiency gains belong to the paper's experiments; they are not a guaranteed
speedup for every architecture or hardware configuration.

<h2 id="scaling-laws">4. Explaining scaling-law exponents</h2>

A measured relation such as

$$
\mathcal{E}(C) \approx \mathcal{E}_{\infty} + A C^{-\alpha}
$$

summarizes how error changes with compute $C$. A mechanism should also explain
what determines $\alpha$, and when the relation changes.

In **How Feature Learning Can Improve Neural Scaling Laws**, Bordelon, Atanasov
and Pehlevan analyze a solvable model beyond the fixed-kernel regime. The outcome
depends on the target's difficulty relative to the initial kernel. For hard
targets outside its reproducing kernel Hilbert space—the function space
associated with that kernel—feature learning can improve the training-time and
compute exponents. For easier targets, the exponents need not change.
[ICLR 2025; preprint first posted in 2024](https://arxiv.org/abs/2409.17858).

**What advances beyond fitting curves:** representation learning can change the
rate at which extra compute pays off. The analytical result is established in
a simplified model, with supporting experiments on nonlinear networks. It does
not identify one universal exponent for language-model training.

<h2 id="optimization">5. Optimization at the edge of stability</h2>

For gradient descent on a positive-curvature quadratic, stability requires the
step size to be sufficiently small. Along a direction with curvature $\lambda$,
the relevant threshold is $\eta\lambda=2$.

Neural-network training often approaches this boundary:

$$
\eta\,\lambda_{\max}(H) \approx 2,
$$

where $\eta$ is the learning rate and $H$ the loss Hessian. Loss can rise during
individual steps as the parameters oscillate, even while longer-term training
makes progress. Changes in curvature can feed back on those oscillations.

**Understanding Optimization in Deep Learning with Central Flows**, by Cohen,
Damian and colleagues, develops differential equations for the time-averaged
trajectory of oscillatory optimizers. Their predictions capture long-term
trajectories numerically and explain how the dynamics interact with the local
loss geometry.
[ICLR 2025 paper and illustrated explanation](https://centralflows.github.io/).

**Connection to effective theory:** averaging over fast oscillations exposes
slower dynamics that small-step gradient flow misses. The strongest demonstrations
concern deterministic optimization; realistic stochastic training introduces
additional questions. The threshold above is for ordinary gradient descent,
not a universal stability formula for every optimizer.

<h2 id="in-context">6. When depth helps in-context learning</h2>

In-context learning adapts predictions using examples supplied in the input,
without updating model weights. Explaining which resources make it effective
requires separating width, depth, context length and training time.

Bordelon, Letey and Pehlevan do this for regression in a deep linear
self-attention model. Under isotropic or fixed-covariance task settings, their
analysis finds that depth helps only when context length is limited. When
structured covariances vary across contexts through random rotations, depth
can remain helpful even with unlimited context. Their joint asymptotic analysis
connects risk to model shape and computational resources.
[Theory of Scaling Laws for In-Context Regression, ICLR 2026; preprint first posted in October 2025](https://proceedings.iclr.cc/paper_files/paper/2026/hash/24d36eee157559e0d2549455fba28f6a-Abstract-Conference.html).

**Why it matters:** the benefit of depth is linked to the statistical structure
of tasks. This is a solvable model of in-context regression, not an explanation
of all language-model reasoning.

<h2 id="reading">What to read next</h2>

The following map connects the questions above to a manageable reading path.

| If you want to understand… | Start with… | Keep track of… |
| --- | --- | --- |
| The closest continuation of the book | [Hanin–Zlokapa](https://doi.org/10.1103/xm8d-v66z) | Bayesian inference, shaped nonlinearities and the joint limits |
| Strong feature-learning dynamics | [Bordelon–Pehlevan DMFT](https://arxiv.org/abs/2205.09653) | Parameterization and evolving collective quantities |
| Practical consequences for training | [CompleteP](https://arxiv.org/abs/2505.01618) | Depth scaling, optimizer settings and the experimental scope |
| Why scaling exponents change | [Feature learning and scaling laws](https://arxiv.org/abs/2409.17858) | Target difficulty relative to the initial kernel |
| Oscillatory optimization | [Central flows](https://centralflows.github.io/) | Finite learning rates and deterministic versus stochastic dynamics |
| Why transformer depth helps | [In-context regression scaling](https://arxiv.org/abs/2510.01098) | Task covariance, context length and linear attention |

For a broader map, Simon and colleagues' April 2026 perspective
**[There Will Be a Scientific Theory of Deep Learning](https://arxiv.org/abs/2604.21691)**
brings together solvable models, tractable limits, macroscopic laws,
hyperparameter theories and recurring phenomena under the proposed name
*learning mechanics*. It is a research perspective and guide to open problems,
rather than a completed unified theory.

My reading of these results is that the field is making progress toward
quantitative explanations of the learning process. The remaining challenge is
to connect those explanations across regimes: from controlled models to finite
networks, from training dynamics to generalization, and from simplified tasks to
the representations and algorithms learned by pretrained models.

Related on this blog: the
[Random Matrices Meet Deep Learning series roadmap](rmt-deep-learning-series.html)
and the [Free Probability & Deep Learning draft](free-probability-deep-learning.html)
develop the spectral tools behind several of these questions.
