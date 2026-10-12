# a01 NAFNet U-Net

**How it works:** U-Net of NAFBlocks (LayerNorm → 1x1/depthwise conv → SimpleGate (channel split-multiply) → simplified channel attention → 1x1 FFN, no nonlinear activations). Predicts the residual (clean − corrupted). Width 32, enc [2,2,4,8], mid 4, dec [2,2,2,2]. Loss: Charbonnier. 60 epochs.
**Cost:** 12.8 GFLOPs, 14.3M params.

## Results
Round 1 (old pipeline, not comparable with round 2): PSNR 27.32, SSIM 0.7522, DISTS 0.2280, 12.82 GFLOPs.
