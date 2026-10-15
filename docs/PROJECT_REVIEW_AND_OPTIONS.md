# Project review and options for the generative model

Written 2026-10-12. Purpose: one place to understand what has been built and tested so far, and what the realistic options are for a strictly generative model, so the next plan can be decided with full information.

How to read the sources: every idea in the "can be explored" sections names its paper right next to it. A link means I checked the paper online while writing this. "(from memory)" means the paper is well known to me but I did not re-check venue and year today, so confirm before citing it in the report. Repo facts were read from the code and result files, not assumed.

---

> **Update 2026-10-15 (after rounds 4–5).** Options tested: A (a20, a21, a24, a25), B (a22, a23), D (a26), wavelet discriminator §5.3 (a21, a25). Outcome: A is the only viable family at ≤ 5 GFLOPs; B and D are closed (behind A on every metric, more steps always worse). Making A really sample cost ~1 dB with the round-5 recipe. New options H–K (§3, after Option G) address that cost. Results: `COMPARISON.md` §9–10.

## 0. The task in short

From `Mini Competition.pdf`:

- Build a **small generative model** that takes one corrupted 256×256×3 RGB image and returns one clean 256×256×3 **uint8** image.
- It must be general: nine named corruption types, their mixtures, and **possibly other types not disclosed**. Test images are natural photos, low-light photos and medical images (X-ray, MRI, etc.), about 100 in total.
- Four numbers are ranked separately across all teams: **FLOPs** (lower is better, measured with fvcore/ptflops on one call with a 1×3×256×256 tensor), **PSNR**, **SSIM** (higher is better) and **DISTS** (lower is better).
- Deliver weights, inference code and a 1–2 page report (architecture, training objective, degradations).

Two consequences that shape everything below:

1. Every sampling step of a generative model runs inside the one measured call, so **K steps cost K times the FLOPs**.
2. PSNR, SSIM and DISTS all compare the output with the one true clean image. Nothing rewards an output that looks realistic but differs from the truth.

---

## 1. Dataset and noise degradation

**Status: this part is complete and reusable as it is.** Nothing here depends on whether the model is generative.

### 1.1 Clean images

Source: `src/common/dataset.py`. Ten sources, three domains, about 30 GB downloaded once by `scripts/prepare_data.py`.

Natural photographs
- **DIV2K** (800 training photos, 2K resolution) and **Flickr2K** (2,650 photos, 2K): high-quality photos. Each is stored twice: as a whole image shrunk to 512 px, and as native-resolution 512 px tiles.
- **COCO test2017** (about 40,000 everyday photos): variety of content.
- **BSDS500** (500 photos): classic denoising benchmark images.
- **LOL normal-light images**: the well-lit half of the LOL pairs.

Low-light photographs
- **LOL low-light images** (real dark photos).
- **Synthetic darkening**: 10% of natural training samples are darkened on the fly (gamma 1.0–1.5, brightness ×0.08–0.6, optional colour cast). The darkened image is the *clean target*; corruption is added on top.

Medical images
- **NIH ChestX-ray** (about 4,900 X-rays).
- **IXI T2** (brain MRI, volumes cut into slices).
- **MedMNIST OrganC** (CT) and **MedMNIST Breast** (ultrasound), 224 px versions.

How training samples are drawn: by weight per source, not by dataset size, so the 40,000 COCO images cannot drown out the small ultrasound set. Weights: natural 70% (COCO 26, whole DIV2K/Flickr2K 22, tiles 16, BSDS 4, LOL normal 2), low-light 3% real plus the synthetic darkening, medical 20% (X-ray 8, MRI 7, CT 3, ultrasound 2). Each sample: resize the short side to roughly 256–512 px, random 256×256 crop, random flips and 90° rotations, 4% converted to grey.

Held-out images: separate images from BSDS, COCO val2017, LOL, X-ray, MRI (30 held-out subjects), CT and ultrasound. They are never used for training.

### 1.2 Corruptions

Source: `src/common/degradations.py`. There are 26 corruption functions. Each takes a severity `s` from 0 (mild) to 1 (severe).

The nine families named in the brief (always trained):
- Gaussian noise (σ 2–55 on a 0–255 scale; 25% of the time the same noise on all three channels)
- Poisson noise (1000 down to 5 photons at white)
- Salt-and-pepper (0.2%–20% of pixels)
- Speckle noise (σ 0.03–0.45, multiplicative)
- Gaussian blur (σ 0.4–4 px)
- Motion blur (length 3–25 px, straight or shaky path)
- Defocus blur (radius 0.8–8 px)
- JPEG (quality 95 down to 8)
- Chromatic aberration (colour planes shifted 0.5–6 px)

Eight extra families trained from round 2 ("v1" pipeline): correlated noise, random-valued impulse noise, Rician noise (MRI-like), stripe noise, resize blur, WebP compression, colour quantisation, small artifacts (1–24 small blobs/scratches).

Five more trained from round 3 ("v2" pipeline): pixelation, periodic noise, sensor noise (shot + read noise), ringing (over-sharpening), double JPEG.

Never trained, kept only for testing: uniform noise (and, for v2-trained models, film grain, glass blur, line dropout).

How a training corruption is built (`RandomDegradation`):

```
clean image
   |
   v
choose how many corruptions: 0, 1, 2, 3 or 4
   (v2 probabilities: 8%, 34%, 36%, 17%, 5%)
   |
   v
choose which ones (65% of picks from the nine listed families,
   at most 1 blur, 1 compression, 1 impulse, 2 noises per chain)
   |
   v
choose a severity for each (Beta(1.0, 1.8): mostly mild; milder still in long chains)
   |
   v
apply in camera order: optics -> blur -> noise -> impulse -> compression
   (50% of chains in random order instead)
   |
   v
round to 8-bit  ->  corrupted input
```

The 8% "zero corruptions" case teaches the model to leave a clean image alone.

### 1.3 Testing (fixed benchmarks)

Built once and saved to disk, so every model is scored on exactly the same corrupted images.

- **Benchmark v1: 396 images, 33 cases, 12 images per case (6 natural, 3 low-light, 3 medical), severities spread evenly from 0.1 to 0.9.** Average input PSNR 24.49 dB. Groups: *single* (the nine listed families), *extra* (correlated noise, resize blur, small artifacts), *pair* (12 two-corruption mixes), *triple* (5 three-corruption mixes), *unseen* (uniform noise, pixelate, periodic noise, pixelate+uniform noise).
- **Benchmark v2: 96 images, 8 cases.** *unseen_v2*: film grain, glass blur, line dropout, glass blur+film grain, line dropout+JPEG (no model has trained on these). *plan*: sensor noise, ringing, double JPEG.

### 1.4 What we learned about data (still valid)

- Adding a family to training improves that family by 3–8 dB. Families never seen improve by only about 0.1 dB. **The model restores what it has been shown.** So the list should keep growing.
- Our benchmark uses our own corruption code. The organisers will use theirs. Our numbers are optimistic estimates.

---

## 2. Models explored

### 2.1 What was trained

Nineteen configurations. Rounds 2 and 3 use the same benchmark, about 5 GFLOPs and 60k training steps unless stated, so they compare directly. Round 1 used an older DIV2K-based pipeline and a different validation set; its numbers are not comparable with the rest.

Round 1 (old pipeline)
- **a01 NAFNet U-Net** (regression): 27.32 dB, 12.8 GFLOPs.
- **a02 Restormer-lite** (transformer, regression): 27.44 dB, 11.3 GFLOPs.
- **a03 Pix2Pix GAN** (**generative**): a01's network as generator plus a 3-layer PatchGAN discriminator, L1×100 + adversarial loss. 26.54 dB, worse than a01 on all metrics.
- **a04 small conditional DDPM** (**generative**): diffusion from pure noise, 8 steps, noise prediction. Collapsed (10 dB) at 37 GFLOPs.

Round 2 (seven architectures at ~5 GFLOPs)
- **a05 NAFNet**: 31.92 dB / 0.8550 SSIM / 0.1542 DISTS. Best of the round.
- **a06 Restormer**: 31.68 dB. Worse than NAF at equal cost.
- **a07 NAF + global attention bottleneck**: 31.85 dB.
- **a08 a07 + degradation-type embedding**: 31.84 dB.
- **a09 NAF in the wavelet domain**: 31.89 dB.
- **a10 residual flow, 2 steps** (flow-style, but **no noise input**, so deterministic): 30.35 dB, worst.
- **a11 NAF + FFT branch**: 31.84 dB.

Round 3 (one change at a time on a05)
- **a12 / a13 / a14 smaller models** (2.3 / 1.3 / 2.5 GFLOPs): −0.49 / −1.05 / −0.57 dB.
- **a15 wider corruption set (v2)**: 32.03 dB, big gains on the added families.
- **a16 a15 + attention**: no gain, worse on unseen corruptions.
- **a17 / a18 perceptual loss earlier / stronger**: slightly better DISTS, lower PSNR and SSIM.
- **a19 = a15 trained 200k steps**: **33.11 dB / 0.8743 / 0.1293 at 4.98 GFLOPs**. Best model, but regression.

### 2.2 What the results say

1. **Architecture details hardly matter at equal FLOPs.** The top five of round 2 are within 0.09 dB. Attention, FFT, wavelets, Restormer blocks and degradation labels gave nothing measurable.
2. **Training length matters most** (+1.08 dB from 60k to 200k steps), **then corruption coverage.**
3. **Size has a clear price:** halving FLOPs costs about 0.5 dB, quartering about 1 dB.
4. **Splitting the FLOPs across several steps is very expensive.** a10 used two half-size passes and scored below a13, which is a single pass at a quarter of the cost.
5. **Of 19 models, only a03 and a04 are truly generative, and both were weak versions of their family:**
   - a03 trained the GAN from scratch with only an L1 loss, a small discriminator and no perceptual loss.
   - a04 started from pure noise, which throws away the input image, and used only 8 steps.
   
   So we have **no fair measurement yet** of a well-built GAN or diffusion model on this task.

### 2.3 Which type of model has the best scope for the four metrics

A regression network outputs the *average* of all plausible clean images. That average has the best possible PSNR. A generative model outputs *one plausible* clean image. In theory an exact sampler has up to 3 dB lower PSNR than the average (Blau & Michaeli, "The Perception-Distortion Tradeoff", CVPR 2018, from memory). The gap is small when the answer is nearly certain (light noise, small artifacts) and large when information is destroyed (heavy blur plus noise).

So for our scoring the best generative model is one that:
- runs **one step, or very few** (FLOPs),
- **starts from the corrupted image**, not from pure noise (keeps what is already known, so fewer steps and higher PSNR),
- adds **only a little randomness** (PSNR, SSIM),
- produces **realistic texture** where the average would be blurry (DISTS).

Ranking of the families by fit, with reasons:

- **Conditional GAN (U-Net generator + adversarial loss): best fit on FLOPs.** One pass, the discriminator is used only in training and costs zero inference FLOPs. Risk: unstable training, invented detail that lowers PSNR. Our own LLM-prepared review (`Generative_Denoising_Literature_Review.pdf`, §3.1 and §7) also lists this as an implementable lightweight option; that file is our reference note, not guidance from the instructor.
- **Diffusion / flow that starts at the corrupted image, 1–4 steps: best fit on quality, most "textbook generative".** Risk: every extra step multiplies FLOPs.
- **Diffusion + GAN combined:** a diffusion model whose large steps are trained with a discriminator. Gets diffusion's stability and GAN's few-step sharpness.
- **VAE:** cheap and single-pass, but outputs are blurry; weak on all three quality metrics on its own. Useful only as a component (for example to give a GAN a meaningful noise input).
- **Transformer as a generative model** (autoregressive or masked-token): many passes per image, far too many FLOPs. A transformer as a *backbone* inside a GAN or diffusion model is possible, but our a06 showed it loses to NAF blocks at equal FLOPs.
- **Normalising flows:** must be invertible at full resolution, which makes them heavy. Poor fit.
- **Pretrained Stable-Diffusion-based restorers:** hundreds of GFLOPs or more. Not possible in a "small model" competition.

```
                     strictly generative?    FLOPs      PSNR/SSIM     DISTS
Regression (a19)            no                5           best          good      <- not allowed
Conditional GAN             yes               5           slightly      can be
                                                          lower         better
Diffusion from input,       yes               5 x K       lower as      better as
  K = 1..4 steps                                          K grows       K grows
Diffusion + GAN, 1-2 steps  yes               5 or 10     between       good
VAE                         yes               ~5          low           poor
Diffusion from pure noise   yes               5 x 50+     low           good      <- too expensive
Pretrained latent diffusion yes               100s        low           good      <- too expensive
```

(The quality entries for the generative rows are expectations from the literature, not our measurements.)

---

## 3. Models that can be explored

What our own LLM-prepared review suggests (`Generative_Denoising_Literature_Review.pdf`, §7; it is not from the instructor, so treat it as a starting list, not as a rule): a NAFNet-style small U-Net; feed the corrupted image as conditioning (SR3 style); keep it single-pass, "one denoising step of a diffusion model"; alternatives: Pix2Pix-style conditional GAN, few-step latent refinement with a small VAE, a frequency-aware branch (AdaIR), and a pretrained diffusion prior only if heavily distilled. The options below follow that list and extend it.

Each option says: the idea, the source, the FLOPs situation, and how to make it cheap if it is not.

### Option A. Conditional GAN with a NAF U-Net generator

- **Idea.** Generator = our NAF U-Net with a noise input. A discriminator looks at (corrupted, output) pairs and judges real vs generated. Train with reconstruction losses plus an adversarial loss.
- **Sources.**
  - Pix2Pix: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks", CVPR 2017 (in our review PDF, §3.1). The basic recipe: U-Net generator + PatchGAN + L1.
  - Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data", ICCVW 2021 (from memory). Two things to copy: (1) a **U-Net discriminator with spectral normalisation**, which gives per-pixel feedback and is stable on mixed corruptions; (2) train the generator for reconstruction first, then switch the adversarial loss on.
  - Karras et al., "A Style-Based Generator Architecture for GANs" (StyleGAN), CVPR 2019 (from memory): inject noise into each layer so the randomness produces fine texture. This is how to make the noise input actually matter.
  - Liang et al., "Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution" (LDL), CVPR 2022 ([paper page](https://mlanthology.org/cvpr/2022/liang2022cvpr-details)): how to stop a GAN from inventing false detail (see §7).
- **FLOPs.** About 5 GFLOPs, one pass. The discriminator adds nothing at inference. A width-16 generator would be 2.3 GFLOPs.
- **Why a03 failed and this should not.** a03 had adversarial loss from step 0, a weak discriminator, L1 only, old data. The fixes above are all standard.
- **Risk.** PSNR/SSIM drop if the adversarial weight is too high; artifacts on medical images.

### Option B. Residual diffusion: start from the corrupted image, few steps

- **Idea.** Classic diffusion (DDPM, SR3) starts from pure noise and needs 50–1000 steps. Instead, define the noising process so that it ends at "corrupted image + a little noise". The reverse process then only has to remove the *difference* between corrupted and clean, which takes 1–15 steps.
- **Sources.**
  - Yue et al., "ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting", NeurIPS 2023 ([code](https://github.com/zsyOAOA/ResShift)); extended version "Efficient Diffusion Model for Image Restoration by Residual Shifting" ([summary](https://fugumt.com/fugumt/paper_check/2403.07319v3)). Reports strong results with as few as 4 steps. This is the closest template for us.
  - Delbracio & Milanfar, "Inversion by Direct Iteration (InDI)", TMLR 2023 (from memory): straight path from corrupted to clean; shows that 1 step gives the "average" answer and more steps give more realistic detail.
  - Liu et al., "Residual Denoising Diffusion Models (RDDM)", CVPR 2024, and Zheng et al., "Selective Hourglass Mapping for Universal Image Restoration Based on Diffusion Model (DiffUIR)", CVPR 2024 (both from memory): the same idea applied to several corruption types with one model, which is our setting.
  - SR3: Saharia et al., "Image Super-Resolution via Iterative Refinement" (our review PDF, §4.3): the conditioning trick, concatenating the corrupted image to the network input at every step.
- **FLOPs.** 5 GFLOPs × number of steps with our current network. To stay near 5 GFLOPs in total: 1 step at full size, or 2 steps at width 16 (4.6 GFLOPs), or 4 steps at width 12 (5.3 GFLOPs).
- **Risk.** With 1 step and a clean copy of the input, the network may ignore the noise and behave like regression. This must be measured (do different random seeds give different outputs?). More steps fix it but cost FLOPs.

### Option C. Distil a multi-step diffusion model into one step

- **Idea.** Train the multi-step model of Option B as a "teacher", then train a one-step "student" to reproduce the teacher's output from the same noise. The student is still a generative sampler, at the FLOPs of one pass.
- **Sources.**
  - Wang et al., "SinSR: Diffusion-Based Image Super-Resolution in a Single Step" (reported as CVPR 2024; venue not re-checked): one-step distillation of ResShift.
  - "One-Step Residual Shifting Diffusion for Image Super-Resolution via Distillation (RSD)" ([arXiv 2503.13358](https://arxiv.org/pdf/2503.13358)): a newer ResShift distillation that the authors report beats SinSR and even the teacher on perceptual metrics.
  - Song et al., "Consistency Models", ICML 2023 (from memory): the general method for one-step sampling from a diffusion model.
- **FLOPs.** 5 GFLOPs at inference. Training cost is roughly doubled (teacher + student).
- **Risk.** Two training stages take time; only worth it if Option B at 2–4 steps is clearly better than at 1 step.

### Option D. Diffusion + GAN in one model

- **Idea.** A normal diffusion model assumes each denoising step is small. With very few steps each step is large, and that assumption breaks. Fix: let a conditional GAN model each large step.
- **Source.** Xiao, Kreis & Vahdat, "Tackling the Generative Learning Trilemma with Denoising Diffusion GANs", ICLR 2022 Spotlight ([paper page](https://iclr.cc/virtual/2022/poster/7183), [code](https://github.com/XavierXiao/denoising-diffusion-gan)): generation in as few as two steps.
- **FLOPs.** 5 GFLOPs at 1 step, 10 at 2 steps with our network; the discriminator is training-only.
- **Why it is attractive.** It is Option A and Option B in a single model: clearly a diffusion model, with the adversarial loss forcing the noise to be used.

### Option E. Do the generative work at lower resolution (wavelet or latent)

- **Idea.** If several steps are needed, run them on a smaller grid so each step is cheaper.
- **Sources.**
  - Huang et al., "WaveDM: Wavelet-Based Diffusion Models for Image Restoration", IEEE Transactions on Multimedia ([arXiv 2305.13819](https://arxiv.org/pdf/2305.13819)): diffusion on wavelet coefficients, about 5 sampling steps, tested on several tasks including denoising.
  - Jiang et al., "Low-Light Image Enhancement with Wavelet-based Diffusion Models (DiffLL)", SIGGRAPH Asia 2023 ([arXiv 2306.00306](https://arxiv.org/pdf/2306.00306)): diffusion only on the low-frequency wavelet band, plus a light module that restores high frequencies.
  - Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022 (our review PDF, §4.4): the same principle with a VAE instead of wavelets.
- **FLOPs.** One Haar transform turns 256×256×3 into 128×128×12 with no loss. A NAF block costs four times less there, so 2 steps at 128 px cost about the same as 1 step at 256 px with the same width. Two transforms give 64×64×48 and 16 times cheaper blocks.
- **What we already know.** Our a09 ran a regression NAF U-Net at 128 px in the Haar domain and matched a05, so half-resolution processing loses nothing for the network body. Its weakness was salt-and-pepper noise. A learned VAE would lose fine medical detail and needs extra training; the wavelet version is free and exact, so it is the better choice here.

### Option F. Flow matching instead of diffusion

- **Idea.** Learn a straight-line "velocity" from (corrupted + noise) to clean; sample with a few Euler steps. Simpler maths than diffusion, same network.
- **Sources.**
  - Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023, and Liu et al., "Flow Straight and Fast: Rectified Flow", ICLR 2023 (both from memory).
  - Zhu et al., "FlowIE: Efficient Image Enhancement via Rectified Flow", CVPR 2024 ([paper](https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_FlowIE_Efficient_Image_Enhancement_via_Rectified_Flow_CVPR_2024_paper.html)): restoration in fewer than 5 steps. Note it relies on Stable Diffusion, so only the training idea transfers, not the model.
  - Ohayon et al., "Posterior-Mean Rectified Flow (PMRF)", ICLR 2025 ([arXiv 2410.00418](https://arxiv.org/abs/2410.00418v3)): first predict the average image, then use a flow to move it to a realistic image. Good theory, but it needs two networks (two sets of FLOPs) and its first stage is a regression model.
- **FLOPs.** Same as Option B. In practice B and F are two ways of writing nearly the same model; choose one, not both.

### Option G. VAE as a helper, not as the model

- **Idea.** A small encoder turns the image into a random latent vector; the generator uses it as its noise. Gives the GAN of Option A a meaningful, trainable source of randomness.
- **Sources.** Prakash et al., "Fully Unsupervised Diversity Denoising with Convolutional Variational Autoencoders (DivNoising)", ICLR 2021 (from memory): a VAE that produces diverse denoised samples. Zhu et al., "Toward Multimodal Image-to-Image Translation (BicycleGAN)", NeurIPS 2017 (from memory): VAE + conditional GAN so that the noise is not ignored.
- **FLOPs.** A latent-vector encoder at low resolution adds a few percent.
- **Verdict.** Only if Option A's noise turns out to be ignored.

### Status of A–G after rounds 4–5

| Option | Tested as | Result | Status |
|---|---|---|---|
| A. Conditional GAN | a20, a21 (round 4), a24, a25 (round 5) | stable, no medical/low-light artifacts; noise ignored unless forced; forced → −1 dB | **main family** |
| B. Residual diffusion | a22 (1 step), a23 (2 steps, w16) | −0.52 / −2.1 dB vs a15; more steps worse | closed |
| C. Distillation | – | needs a good multi-step teacher, which B did not give | dropped |
| D. Diffusion + GAN | a26 | −0.27 dB vs a24, noise mostly harmful | closed |
| E. Wavelet / latent steps | – | only useful for multi-step, which is closed | dropped |
| F. Flow matching | – | same situation as B | dropped |
| G. VAE latent noise | – | not needed: round 5 shows the noise can be forced to matter | held back |

### Option H. Variance-controlled cGAN: full per-sample loss + diversity constraint (round 6, a27)

- **Idea.** The round-5 recipe applied the reconstruction loss to the mean of two samples. That halves each sample's reconstruction gradient, and the outputs became over-smoothed (−0.75 dB from a worse average output, only −0.25 dB from the variance itself). Instead, keep the composite loss on **every** sample at full weight, as in a20, which had the best GAN quality. Buy the variance with a **constraint**: a diversity hinge, active, on a few extra noise draws per batch. The model then pays exactly the minimum variance needed, and the batch stays at 16 images.
- **Sources.**
  - Bendel, Ahmad & Schniter, "A Regularized Conditional GAN for Posterior Sampling in Image Recovery Problems" (rcGAN), NeurIPS 2023 ([paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d8b29f07599fecdba93d87ed27a65524-Abstract-Conference.html), [code](https://github.com/matt-bendel/rcGAN)). An ℓ1 penalty plus an **adaptively weighted standard-deviation reward**; the weight is tuned during training so that the sample variance matches a target. Our hinge with a target is a simple form of the same control.
  - Ohayon, Adrai, Vaksman, Elad & Milanfar, "High Perceptual Quality Image Denoising with a Posterior Sampling CGAN", ICCVW 2021 ([arXiv 2103.04192](https://arxiv.org/abs/2103.04192)). A penalty on the **mean of the samples** (first-moment penalty) instead of each sample: this is our round-5 recipe. It is designed for perceptual quality and accepts the distortion cost, which matches what we measured.
  - Blau & Michaeli, "The Perception-Distortion Tradeoff", CVPR 2018 ([paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html)). Lower distortion and better perceptual quality conflict, which is why the scored PSNR/SSIM push towards the least random model that is still eligible.
- **FLOPs.** None at inference (training-only change). Training cost +25% (16 images + 4 extra draws).
- **Risk.** At full per-sample weight the noise is penalised again; the hinge has to win. Monitor `train_spread255`.

### Option I. Two-stage training: reconstruction first, then adversarial + diversity (round 6, a28)

- **Idea.** Train the noisy generator with the composite loss only for the first half (this is what a20 effectively became), then switch on the adversarial and diversity terms for the second half. This is the standard recipe of the GAN restorers that keep high PSNR.
- **Sources.** Ledig et al., "SRGAN", CVPR 2017; Wang et al., "ESRGAN", ECCVW 2018; Wang et al., "Real-ESRGAN", ICCVW 2021 (all from memory). Each pre-trains a PSNR-oriented network, then fine-tunes it with GAN losses. ESRGAN also interpolates the weights of the two stages to dial the trade-off (**network interpolation**, zero extra training).
- **FLOPs.** None at inference.
- **Trade-off with our rules.** The generator has its noise input from step 0, but the generative loss starts at 50%, so this relaxes plan decision "generative loss from the first step". The final model is still a GAN trained adversarially that samples. Use it only if a27 cannot close the gap.

### Option J. Test-time noise scale (temperature)

- **Idea.** Scale the noise at inference to trade diversity for quality, as SRFlow does with its sampling temperature (Lugmayr et al., "SRFlow", ECCV 2020, from memory).
- **Measured (round 5).** For the GANs ×0.5 changes PSNR by ≤ 0.05 dB, improves DISTS by ~0.003 and keeps diversity ≥ 1.1. **Adopted**: free, choose per final model.

### Option K. Normalising-flow restorer with temperature (not now)

- **Idea.** SRFlow-style conditional flow: exact likelihood, sampling with a temperature; at low temperature it is close to the average answer.
- **Why not now.** New architecture family, invertible layers are FLOP-heavy at 256 px, and there is no time to tune it within the deadline. Noted for the report as related work.

### Not worth exploring, with reasons

- **Zero-shot diffusion priors** (DDRM, DiffPIR; our review PDF §4.2 and §4.5): need a known corruption formula and 20–100+ steps of a large pretrained model. Our corruptions are unknown and the FLOPs are far too high. The review itself says they are "more useful as inspiration".
- **Pretrained Stable-Diffusion restorers** and **Defusion** (review §4.7): too heavy.
- **Autoregressive / masked-token transformers:** one pass per token group; too heavy.
- **Degradation prompts** (AirNet, PromptIR; review timeline): our a08 tested the idea and it did not help.
- **GenDeg** (review §4.6): its lesson, widen the corruption pipeline, is already our strongest finding and is applied in §1.

### Summary flow

```
                 corrupted image (+ noise)
                          |
        +-----------------+-------------------+
        |                                     |
   one pass                             several passes
        |                                     |
  A. conditional GAN                  B/F. residual diffusion / flow
  D. diffusion-GAN (1 step)                   |
        |                         too many FLOPs?  yes
        |                                     |
        |                    +----------------+----------------+
        |                    |                                 |
        |          C. distil to one step            E. run steps at 128 px
        |                    |                         (wavelet domain)
        +--------------------+---------------------------------+
                          |
              check: noise changes the output?
              compare PSNR / SSIM / DISTS / FLOPs
```

---

## 4. Frequency domain: what we have explored

Three uses so far, all inside regression models.

- **FFT as a training loss (all models a05–a19).** `FFTLoss` in `src/common/losses.py`: the L1 difference between the 2-D Fourier transforms of output and target, weight 0.05. Its job is to punish missing high frequencies (over-smoothing). It costs nothing at inference. It was never tested on its own (never switched off in a controlled run), so its individual effect is unknown.
- **FFT inside the network (a11).** `FrequencyBranch` / `DualDomainBlock` in `src/common/layers.py`: at 128 px and below, each block has a normal spatial path **and** a branch that applies an FFT, filters the spectrum with 1×1 convolutions and transforms back. Full resolution keeps plain NAF blocks. So this is *spatial + frequency together*, not frequency alone. Result: 31.84 dB vs 31.92 for the plain network. It helped only on periodic noise when that noise was unseen (31.6 vs 30.4 dB); the expected gain on blur did not appear. One more issue: FLOPs counters may not count FFT operations, so its true cost is uncertain.
- **Haar wavelet around the network (a09, a14).** `haar_dwt` / `haar_iwt`: the image is transformed once to 128×128×12, a NAF U-Net works on those coefficients, and the result is transformed back. So this is *wavelet-domain only* for the network body. Result: a09 tied a05 at 5 GFLOPs (31.89 dB) and was best on blur and medical images; a14 at 2.5 GFLOPs was slightly worse than the plain small model a12. Weak point: salt-and-pepper noise (−1.3 dB), because one bad pixel spreads into all four wavelet bands.

Not explored at all: frequency ideas in a generative model, wavelet-domain losses, a frequency-aware discriminator, DCT (the JPEG transform), multi-level wavelets, and separate treatment of low and high bands.

---

## 5. Frequency domain: what can still be explored

Our LLM-prepared review points to one idea here: a "frequency-aware branch (AdaIR-inspired) ... since many unseen corruptions still behave predictably in frequency space" (§7). A warning from our own results first: a11 already added a frequency branch to a regression network and gained nothing, so a branch on its own is unlikely to be the answer. The more promising uses are the ones that make a *generative* model cheaper or more stable. Ordered by expected value for us.

### 5.1 Wavelet-domain generative model (cheaper steps)

- **Idea.** Run the diffusion/GAN on Haar coefficients at 128 px. Each step costs about a quarter, so 2 steps fit in the budget of 1.
- **Sources.** WaveDM ([arXiv 2305.13819](https://arxiv.org/pdf/2305.13819)); DiffLL ([arXiv 2306.00306](https://arxiv.org/pdf/2306.00306)); Phung et al., "Wavelet Diffusion Models are Fast and Scalable Image Generators", CVPR 2023 (from memory; I could not re-confirm the title online today), which applies the wavelet trick to the Denoising Diffusion GAN of Option D.
- **FLOPs.** Lower per step; the transform itself is free. We already have the code (`haar_dwt`, a09).
- **Fits with.** Options B, D, E.

### 5.2 Diffuse only the low band, restore the high bands in one shot

- **Idea.** Most of the image content is in the low-frequency band; noise and fine detail are in the high bands. Run the multi-step generative model only on the small low band, and use a light one-pass module for the high bands.
- **Source.** DiffLL ([arXiv 2306.00306](https://arxiv.org/pdf/2306.00306)): wavelet-based conditional diffusion plus a "high-frequency restoration module".
- **FLOPs.** With two Haar levels the low band is 64×64, 16 times fewer pixels than the image, so even 4 steps there are cheap.
- **Caution.** Splits the model into two parts; more code and more to tune.

### 5.3 Wavelet-domain discriminator and losses for a GAN

- **Idea.** Let the discriminator see only the high-frequency wavelet bands, and compute the fidelity loss on wavelet bands. The GAN then learns real detail and produces fewer false textures.
- **Source.** Korkmaz, Tekalp & Dogan, "Training Generative Image Super-Resolution Models by Wavelet-Domain Losses Enables Better Control of Artifacts", CVPR 2024 ([paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Korkmaz_Training_Generative_Image_Super-Resolution_Models_by_Wavelet-Domain_Losses_Enables_Better_CVPR_2024_paper.pdf), [code](https://github.com/mandalinadagi/WGSR)). They report a better balance between PSNR-type and perceptual metrics than RGB-domain or Fourier losses.
- **FLOPs.** Zero at inference (training-only).
- **Fits with.** Option A and D. This directly targets our main worry about GANs (losing PSNR to invented detail).

### 5.4 Frequency-guided modulation for unseen corruptions

- **Idea.** Different corruptions damage different frequency bands (noise: high; blur: removes high; JPEG: block frequencies). Split features into low and high parts using the spectrum of the input, and let them modulate each other.
- **Source.** Cui et al., "AdaIR: Adaptive All-in-One Image Restoration via Frequency Mining and Modulation", ICLR 2025 ([arXiv 2403.14614](https://arxiv.org/pdf/2403.14614)), named in our review PDF.
- **FLOPs.** AdaIR sits on a heavy Restormer backbone. A light version, one such module at the 32 px or 16 px level only, would add a few percent.
- **Caution.** Close to what a11 tried. Worth one test at most, and only measured on the unseen-corruption benchmark.

### 5.5 Cheap frequency selection without FFT

- **Idea.** Split features into low and high frequency using average pooling (low = pooled, high = feature − pooled) and re-weight each with learned channel weights.
- **Source.** Cui et al., "Selective Frequency Network for Image Restoration (SFNet)", ICLR 2023 ([paper page](https://iclr.cc/virtual/2023/poster/10743)), and the follow-up "Exploring Richer and More Accurate Information via Frequency Selection for Image Restoration" ([arXiv 2407.08950](https://arxiv.org/pdf/2407.08950)).
- **FLOPs.** Pooling and a few 1×1 weights: nearly free, and counted correctly by FLOPs tools (unlike FFT).

### 5.6 FFT blocks for blur

- **Sources.** Mao et al., "Intriguing Findings of Frequency Selection for Image Deblurring (DeepRFT)", AAAI 2023, and Kong et al., "Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring (FFTformer)", CVPR 2023 (both from memory; both are cited in our a11 code).
- **Status.** This is what a11 implemented. It did not help. Not recommended again.

Suggested order if frequency ideas are used: 5.3 (free, with any GAN) → 5.1 (if more than one step is needed) → 5.5 (cheap add-on) → 5.2 and 5.4 only if time remains.

---

## 6. Losses used so far

Source: `src/common/losses.py` and each model's `config.yaml`.

- **a01, a02:** Charbonnier loss only (a smooth version of L1 on pixels).
- **a03 (GAN):** L1 × 100 + adversarial loss (binary cross-entropy, non-saturating). No perceptual term.
- **a04 (diffusion):** mean squared error on the predicted noise (the original DDPM loss).
- **a05–a16, a19:** the composite loss, one term per scored metric:
  - Charbonnier × 1.0 → PSNR
  - FFT L1 × 0.05 → sharpness (see §4)
  - (1 − SSIM) × 0.2, computed with the same 7×7 window as the competition's skimage SSIM → SSIM
  - DISTS × 0.1, the scored metric itself with its VGG network frozen, switched on for the second half of training and computed on 4 images per batch → DISTS
- **a08:** composite + a small auxiliary loss (smooth-L1 × 0.1) for predicting which corruptions are present.
- **a10 (flow):** the composite loss applied to the predicted clean image at random points on the path.
- **a17:** composite with DISTS from the first step. Result: DISTS −0.0025, PSNR −0.09 dB.
- **a18:** DISTS × 0.2 and SSIM × 0.3 from the first step. Result: DISTS −0.004, PSNR −0.36 dB, and SSIM also went down.

Also present in the code: a direct PSNR loss and plain L1/L2.

What we learned:
- When DISTS switches on, the DISTS score improves fast (0.19 → 0.15 within a few epochs) at almost no PSNR cost. It is the most useful single loss term.
- Making perceptual terms heavier trades PSNR and SSIM for DISTS at a worsening rate. Weight 0.1 is about right.
- All of these are training-only. **No loss adds inference FLOPs.**

Never tried: a modern adversarial setup (only a03's basic one), any diffusion loss other than a04's, wavelet losses, focal frequency loss, LPIPS, gradient/edge loss, artifact-suppression losses.

---

## 7. Losses that can still be explored

All of these act only during training, so **none of them changes FLOPs**. They are grouped by the model they belong to.

### For a GAN (Option A or D)

- **Hinge adversarial loss with a spectral-norm U-Net discriminator.** More stable than a03's cross-entropy PatchGAN and gives per-pixel feedback. Source: Real-ESRGAN (Wang et al., ICCVW 2021, from memory) for the discriminator; hinge loss as used in "Spectral Normalization for GANs" (Miyato et al., ICLR 2018, from memory). Helps: DISTS, training stability.
- **Relativistic adversarial loss.** The discriminator judges "is the real image more realistic than the generated one" instead of real/fake in isolation; gives the generator gradients from real images too. Source: Wang et al., "ESRGAN", ECCV Workshops 2018 (from memory). Helps: sharper textures.
- **Artifact-aware loss (LDL).** Builds a map of where the GAN output is likely an artifact (using local variance of the residual) and penalises those regions more. Source: Liang et al., LDL, CVPR 2022 ([paper page](https://mlanthology.org/cvpr/2022/liang2022cvpr-details), [code](https://github.com/csjliang/LDL)). Helps: keeps PSNR/SSIM while using a GAN. A later paper notes it reduces but does not remove artifacts.
- **Wavelet-domain fidelity and adversarial losses.** See §5.3. Source: Korkmaz et al., CVPR 2024 ([paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Korkmaz_Training_Generative_Image_Super-Resolution_Models_by_Wavelet-Domain_Losses_Enables_Better_CVPR_2024_paper.pdf)). Helps: better PSNR-vs-perceptual balance.
- **Discriminator feature-matching loss.** Match the discriminator's internal features of real and generated images; a stable, learned perceptual loss. Source: Wang et al., "High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs (pix2pixHD)", CVPR 2018 (from memory). Helps: stability, DISTS.
- **R1 gradient penalty on the discriminator.** Prevents the discriminator from becoming too strong. Source: Mescheder et al., "Which Training Methods for GANs do actually Converge?", ICML 2018 (from memory). Helps: stability across our three very different domains.

### For a diffusion / flow model (Option B, C, D, F)

- **Predict the clean image and apply our composite loss to it.** Instead of the plain noise-prediction MSE that a04 used, let the network output the clean image, so Charbonnier, SSIM and DISTS can all be used on it. Source: ResShift ([code](https://github.com/zsyOAOA/ResShift)) predicts the clean image; "Back to Basics: Let Denoising Generative Models Denoise" (Li & He, 2025, from memory) argues for this in pixel space. Helps: all three quality metrics; this is the main fix for a04.
- **Min-SNR loss weighting.** Different noise levels pull the network in conflicting directions; weighting each level by a clamped signal-to-noise ratio balances them. Source: Hang et al., "Efficient Diffusion Training via Min-SNR Weighting Strategy", ICCV 2023 ([paper](https://openaccess.thecvf.com/content/ICCV2023/html/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.html)), reporting 3.4× faster convergence. Helps: training time, which matters with our deadline.
- **Adversarial loss on each denoising step.** Source: Denoising Diffusion GANs, ICLR 2022 ([paper page](https://iclr.cc/virtual/2022/poster/7183)). Helps: quality at 1–2 steps.
- **Distillation loss (student matches teacher's sample).** Source: SinSR; RSD ([arXiv 2503.13358](https://arxiv.org/pdf/2503.13358)); Consistency Models (Song et al., ICML 2023, from memory). Helps: FLOPs (fewer steps).

### For any model

- **Focal frequency loss.** Like our FFT loss, but it automatically puts more weight on the frequencies the model currently gets wrong. Source: Jiang et al., "Focal Frequency Loss for Image Reconstruction and Synthesis", ICCV 2021 ([paper](https://openaccess.thecvf.com/content/ICCV2021/html/Jiang_Focal_Frequency_Loss_for_Image_Reconstruction_and_Synthesis_ICCV_2021_paper.html)); reported to improve VAE, pix2pix and StyleGAN2. Would replace our plain FFT term. Helps: fine detail, DISTS.
- **Wavelet (Haar sub-band) L1 loss.** Penalise errors separately in low and high bands; cheap, and we already have the transform. Source: same CVPR 2024 paper as §5.3. Helps: edges and texture.
- **LPIPS.** A second perceptual loss. Source: Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric", CVPR 2018 (from memory). Lower priority: we already train on DISTS, which is the scored metric.
- **MS-SSIM + L1 mix.** SSIM at several scales combined with L1. Source: Zhao et al., "Loss Functions for Image Restoration with Neural Networks", IEEE Transactions on Computational Imaging 2017 (from memory). Lower priority: the competition scores single-scale SSIM, which we already match exactly.
- **Edge / gradient loss.** L1 on image gradients. Source: Zamir et al., "Multi-Stage Progressive Image Restoration (MPRNet)", CVPR 2021 (from memory). Helps blur cases a little.
- **Averaging-consistency loss for samplers.** Train so that the *average* of several samples stays close to the truth while individual samples stay sharp; keeps PSNR high for a model that really samples. Source: Whang et al., "Deblurring via Stochastic Refinement", CVPR 2022 (from memory) show that averaging samples recovers PSNR; using it as a training term is my suggestion, not from a paper.

Suggested order: clean-image prediction with the composite loss (needed for any diffusion model) → hinge + U-Net discriminator → LDL or wavelet losses to protect PSNR → focal frequency loss in place of FFT → Min-SNR if training is slow.

---

## 8. What is ready and what has to be built

Ready (no work needed)
- All data, both corruption pipelines, both benchmarks.
- NAF U-Net backbone with a time-embedding input (used by a10) and a Haar transform (used by a09).
- Composite loss, DISTS loss, evaluation, comparison and export scripts.
- FLOPs counting over a whole multi-step sampler (done for a10).
- Trainer support for generator/discriminator losses (from a03).

To be built
- A noise input to the generator, and a measurement of whether it changes the output.
- A modern discriminator (U-Net, spectral norm) and hinge loss.
- A residual-diffusion training loop and a few-step sampler with seeded noise inside the model call.
- Optional: wavelet losses, LDL, distillation.

---

## 9. Decisions you need to make

These are the real choices; everything else follows from them.

1. **Which family goes first?**
   - *GAN (Option A)*: fastest to build, one pass, lowest FLOPs risk.
   - *Residual diffusion (Option B)*: most clearly "generative", but must prove it works in 1–2 steps.
   - *Diffusion-GAN (Option D)*: both at once, more code.
   
   With a short deadline, my suggestion is A and B side by side in two short runs, since they share the backbone, and D only if both show problems (A unstable, or B ignoring its noise).

2. **How many steps are acceptable?** One step keeps FLOPs at about 5. Two steps mean either 10 GFLOPs or a half-size network (about −0.5 dB by our size study), unless the steps run in the wavelet domain (§5.1).

3. **May the generative model start from a19's trained weights?** It would save roughly a day of training and likely give higher PSNR. The final model would still have a noise input and a generative loss. If you want the model generative from its first training step, it trains from scratch (about 17–22 h for the long run).

4. **Which frequency and loss ideas to include?** The zero-FLOPs ones (§5.3, §7) are safe to add to whichever family is chosen. Network-side frequency branches (§5.4–5.6) have a weak record in our own tests.

---

## 10. Honest limits of this document

- The expected behaviour of each generative option comes from papers on other tasks (mostly super-resolution and deblurring), not from our benchmark. Until the first runs finish, the PSNR cost of going generative on our data is unknown.
- Papers marked "(from memory)" need their venue and year checked before going into the report.
- Round-1 numbers (a01–a04) come from an older pipeline and should not be compared with later rounds.
