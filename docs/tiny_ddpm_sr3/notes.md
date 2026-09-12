# tiny_ddpm_sr3 — Architecture Notes

## Source
Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic
Models*. NeurIPS 2020. Combined with the conditioning trick from Saharia,
C. et al. (2022). *Image Super-Resolution via Iterative Refinement (SR3)*.

## What problem it solves
DDPM trains a network to predict, step by step, the noise that was added
to an image -- this IS a generative denoiser by construction. SR3 extends
this to be *conditional*: instead of generating a random image from pure
noise, the network is told (via extra input channels) what specific image
it should be restoring toward.

## How it works (plain language)
1. **Forward process (training only)**: take a clean image, add a random
   amount of Gaussian noise according to a random timestep `t` (out of a
   small total `T`, default 8).
2. **Conditioning (SR3-style)**: concatenate the corrupted input image as
   extra channels alongside the noisy image at every step, so the network
   always knows what it's trying to restore toward -- not just "remove
   generic noise."
3. **Reverse process (inference)**: starting from pure random noise,
   repeatedly apply the trained network for `T` steps, each time
   predicting and subtracting a bit of noise, conditioned on the
   corrupted image throughout.
4. A **FiLM (feature-wise linear modulation)** layer injects the
   diffusion-timestep embedding into every block, so the same network
   weights behave differently depending on which step of the process
   it's currently doing.

## Why this is the most experimental of the four
A full DDPM uses ~1000 sampling steps -- far too expensive for this
competition's FLOPs axis (each step is a full network forward pass). This
implementation deliberately uses a **small `T` (default 8)**, trading some
of the generative quality benefit for a manageable total inference cost
(`T x one forward pass`).

**Important:** the FLOPs number reported in `metrics/final_report.json`
under `gflops_per_forward_call` is for **one** step. The true inference
cost is `gflops_per_forward_call x T` -- this is reported explicitly as
`true_total_gflops_at_inference` in the same file so it's never
underreported. Keep an eye on this number specifically when comparing
against the other three (single-forward-pass) architectures.

## Why try this one anyway
- The literature review's Diffusion Era section argues this family has
  the strongest theoretical basis for zero-shot generalization: a
  generative model has an internal notion of "what a realistic clean
  image looks like" and can push outputs toward realism even for
  corruption types it wasn't explicitly trained on (DDRM's core argument,
  reviewed in Section 4.2).
- Useful as an experiment to see whether the iterative-refinement idea
  measurably helps on the held-out set's undisclosed degradations, even
  at a modest FLOPs premium.

## Key config knobs (`config.yaml`)
- `timesteps` (T): fewer steps = cheaper inference but less refinement;
  more steps = better potential quality but `T x` the FLOPs. Start at 8,
  try 4 (cheaper) or 16 (higher quality, watch FLOPs) if time allows.
- `width` / block counts: keep these SMALLER than `nafnet_unet`'s
  equivalents by default, since every extra parameter here gets "paid for"
  T times at inference.

## Things to try if this underperforms
- Reduce `T` further if FLOPs are the bottleneck (this project's config
  already starts conservative at T=8).
- If sample quality is poor/unstable, this is often a sign the small `T`
  schedule needs a different beta schedule shape, or more training epochs
  (diffusion objectives generally converge slower than direct regression
  -- the default config already allocates more epochs, 80 vs 60, for this
  reason).
- If this architecture isn't winning on any axis compared to the other
  three, that is itself a valid and useful finding for your technical
  report: it demonstrates you explored the generative-diffusion framing
  from the literature review and made an informed trade-off decision
  against it for this specific FLOPs-constrained competition.
