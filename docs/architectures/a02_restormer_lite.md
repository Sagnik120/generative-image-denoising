# a02 Restormer-lite

**How it works:** U-Net with Restormer blocks: MDTA (attention across channels, cost linear in pixels) + GDFN (gated depthwise FFN). Width 24, heads [1,2,4]. Charbonnier loss.
**Cost:** 11.3 GFLOPs.

## Results
Round 1 (old pipeline, not comparable with round 2): PSNR 27.44, SSIM 0.7579, DISTS 0.2352, 11.29 GFLOPs.
