# a28 a27 with the adversarial term delayed (two-phase training)

Round 6. Plan: `../PLAN.md` §5.

**How it works:** identical networks, batch and losses to a27 (`a27_gan_wave_varctl.md`). Only the schedule of the adversarial term differs (`gan.adv_start_frac: 0.5`):
- **Phase 1 (steps 0–30k):** composite + wavelet L1 on every output + diversity constraint (1.5/255, weight 4.0). The noise is in use and enforced from step 0. No adversarial term and no discriminator training (also faster).
- **Phase 2 (30k–60k):** a27's full objective: hinge adversarial 0.005 ramped over 5k steps, discriminator hinge + R1. DISTS also starts at 30k (50%, as in every model).
**4.98 GFLOPs, 5.52 M parameters, one pass. Frequency:** wavelet-domain discriminator (phase 2) + wavelet fidelity + FFT term.
**Why it was tried:** SRGAN (Ledig et al., CVPR 2017), ESRGAN (Wang et al., ECCVW 2018) and Real-ESRGAN (Wang et al., ICCVW 2021) all train for reconstruction first and then adversarially; that is how GAN restorers keep high PSNR. a28 − a27 measures whether this beats adversarial training from step 0 on our data.
**Is it generative?** Yes: it samples throughout training (noise input, diversity enforced from step 0) and the final half is adversarial training, which is the distribution-matching objective. The models above are called GANs with no noise input at all.
**What to look for:** PSNR at epoch 12 (end of phase 1, should be close to a20's 31.65 at that point) and the drop or gain after the switch; `train_spread255` at ~1.5 throughout; final PSNR vs a27.
**Data and schedule:** as a27.

## Results
Pending. Fill in from `results/a28_gan_wave_2stage/metrics/`. Preferred over a27 if ≥ 0.2 dB better with diversity ≥ 0.5.
