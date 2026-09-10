# nafnet_unet — Architecture Notes

## Source
Chen, L., Chu, X., Zhang, X., & Sun, J. (2022). *Simple Baselines for Image
Restoration (NAFNet)*. ECCV 2022.

## What problem it solves
Asks whether the complexity in modern restoration networks (Transformers,
elaborate attention, many nonlinearities) is actually necessary, or whether
a much simpler, cheaper design can match the results.

## How it works (plain language)
- A classic U-Net-shaped encoder-decoder with skip connections.
- Each block strips out standard nonlinear activations (ReLU/GELU/Sigmoid)
  entirely and replaces them with a **Simple Gate**: split the feature map
  into two halves along the channel dimension and multiply them together
  elementwise.
- A **Simplified Channel Attention** block (global average pool -> 1x1
  conv -> rescale) stands in for full self-attention, at a fraction of
  the cost.
- This implementation predicts a **residual** by default
  (`predict_residual: true` in config.yaml): the network outputs
  `corrupted_image + learned_correction` rather than the clean image
  directly, which tends to train more stably (the DDPM-style framing
  discussed in the literature review).

## Why it's the recommended starting point
- Lowest FLOPs of the four architectures in this project at comparable
  model capacity (see the literature review's reported comparison: ~505.5
  GFLOPs for NAFNet vs ~1128.9 GFLOPs for Restormer at 512x512).
- Simple, well-understood building blocks -- fastest to iterate on and
  debug.
- The literature review's Section 7 "Recommended starting plan" names this
  exact design (NAFNet-style Simple Gate blocks) as the primary
  recommendation for this competition's FLOPs-conscious scoring.

## Key config knobs (`config.yaml`)
- `width`: base channel count. 32 is a reasonable default; try 16 for an
  even cheaper variant if FLOPs need to come down further.
- `enc_blocks` / `dec_blocks` / `middle_blocks`: how many NAFBlocks per
  stage. More blocks = more capacity and FLOPs.
- `predict_residual`: keep `true` unless you have a specific reason to
  predict the clean image directly.
- `loss`: `charbonnier` (default, robust smooth-L1-like), or `l1` / `l2`.

## Things to try if this underperforms
- Increase `width` or block counts if PSNR/SSIM plateau too early
  (underfitting).
- Decrease them if FLOPs are too high relative to the marginal quality
  gain (diminishing returns).
- Try `predict_residual: false` and compare -- sometimes direct clean-image
  prediction works better for heavy degradations (e.g. very strong blur).
