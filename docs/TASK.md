# Task: final submission package (code, weights, inference, README)

Updated 2026-10-20. The report is written by the teammate; this file covers the code package. **The package is already built** at `final_submission/generative_denoiser/` with neutral names. No internal model names, round numbers or experiment history appear in it.

## 1. What is submitted

```
generative_denoiser/
├── README.md           # what it is, input/output, usage, FLOPs, results
├── requirements.txt    # torch, numpy, Pillow (+ fvcore, ptflops only for FLOPs)
├── inference.py        # load_model(), denoise_image(), command line; noise scale 0.75 built in
├── model.py            # the network only (generator; no training code)
├── config.json         # network size settings
└── model_weights.pt    # trained weights, PyTorch state_dict, 22.8 MB
```

- **Where it came from:** `model_weights.pt` is a copy of `results/final_model/submission/model_weights.pt` (the downloaded export of the final model; EMA weights of the best checkpoint).
- **The code is a clean rewrite of the network:** same layers, same parameter names. It loads these weights strictly (360 tensors) and gives **bit-identical outputs** to the original export, at noise ×1 and ×0.75 (max difference 0.0, checked 2026-10-20).

## 2. Already verified (2026-10-20, on this laptop, clean environment with only torch, numpy, Pillow, fvcore, ptflops)

| Brief | Check | Result |
|---|---|---|
| §2 input one RGB 256×256×3 | `denoise_image` asserts the shape | (128, 128, 3) is rejected with a clear message |
| §2 output 256×256×3, **uint8 [0, 255]**, conversion in code | `denoise_image` clamps, ×255, rounds, casts to uint8 | (256, 256, 3) uint8, values 25–238; command line writes a (256, 256, 3) uint8 PNG |
| §1 it denoises | noisy gradient image (σ = 25) | 20.25 dB → 45.63 dB |
| §1 generative model | new noise per call / fixed seed | fresh calls differ; same seed identical; diversity 0.83 grey levels (8 seeds) |
| §5 FLOPs on (1, 3, 256, 256) with fvcore / ptflops | on the module returned by `load_model()` (noise ×0.75 inside) | **fvcore 4.982 GFLOPs**, ptflops 5.085 GMACs, 5.52 M parameters |
| §7.1 weights in a standard format + inference code | `model_weights.pt` (PyTorch state_dict) + `inference.py` | loads strictly; CPU ~0.04 s per image |

## 3. Commands to run HERE (laptop, in the repo folder)

**3.1 Environment for the checks (once, ~2 min):**

```
python3 -m venv /tmp/denoiser_env
/tmp/denoiser_env/bin/pip install -r final_submission/generative_denoiser/requirements.txt
```

**3.2 Re-run the checks yourself (~1 min):**

```
cd final_submission/generative_denoiser
/tmp/denoiser_env/bin/python - <<'EOF'
import numpy as np, torch
from inference import denoise_image, load_model
rng = np.random.default_rng(0)
clean = np.tile(np.linspace(30, 220, 256, dtype=np.float32)[None, :, None], (256, 1, 3))
img = np.clip(clean + rng.normal(0, 25, clean.shape), 0, 255).astype(np.uint8)
out = denoise_image(img)
assert out.shape == (256, 256, 3) and out.dtype == np.uint8
mse = lambda a: ((a.astype(float) - clean) ** 2).mean()
print("output", out.shape, out.dtype, "| PSNR %.2f -> %.2f dB" % (10*np.log10(255**2/mse(img)), 10*np.log10(255**2/mse(out))))
print("fresh calls differ:", not np.array_equal(denoise_image(img), denoise_image(img)),
      "| same seed identical:", np.array_equal(denoise_image(img, seed=1), denoise_image(img, seed=1)))
from fvcore.nn import FlopCountAnalysis
fa = FlopCountAnalysis(load_model(), torch.rand(1, 3, 256, 256)); fa.unsupported_ops_warnings(False)
print("fvcore GFLOPs: %.3f" % (fa.total() / 1e9))
EOF
/tmp/denoiser_env/bin/python inference.py --input <any 256x256 png> --output /tmp/out.png
rm -rf __pycache__
cd ../..
```

Expected: `output (256, 256, 3) uint8 | PSNR 20.25 -> ~45.6–46.0 dB` (varies slightly, since every call draws new noise), `True | True`, `fvcore GFLOPs: 4.982`.

**3.3 Zip (after §4, so the README numbers are final):**

```
cd final_submission
find generative_denoiser -name "__pycache__" -type d -prune -exec rm -rf {} +
rm -f generative_denoiser.zip && zip -r generative_denoiser.zip generative_denoiser
rm -rf /tmp/zipcheck && mkdir /tmp/zipcheck && unzip -q generative_denoiser.zip -d /tmp/zipcheck
cd /tmp/zipcheck/generative_denoiser && /tmp/denoiser_env/bin/python -c "import numpy as np; from inference import denoise_image; o=denoise_image(np.zeros((256,256,3),np.uint8)); print('zip OK', o.shape, o.dtype)"
cd - && ls -lh generative_denoiser.zip
```

Submit `generative_denoiser.zip` + the teammate's report PDF.

**Git:** do not commit `final_submission/generative_denoiser/model_weights.pt` or the zip (large binaries). If you want the package code in the repo, commit only the five text files:

```
git add final_submission/generative_denoiser/{README.md,requirements.txt,inference.py,model.py,config.json} docs/TASK.md
```

## 4. Commands to run on the GPU machine (only for the report numbers, ~25 min)

These use the internal experiment names. They produce numbers for the teammate's report, not files for the package.

```
cd <repo on the GPU machine>
git pull origin main
source <venv>/bin/activate
export R=/DATA1/shrusti/genai_project/results
export DATA=<DATA>

# 4.1 full results at the submitted noise scale 0.75 (~20 min)
python scripts/evaluate.py --arch a29_tuned --data_root $DATA --results_root $R --noise_scale 0.75
```

Files to give the teammate, all in `$R/a29_tuned/metrics/`:
- `final_evaluation_summary_noise0.75.json` (overall, by group, by domain);
- `benchmark_breakdown_noise0.75.csv` (33 cases);
- `benchmark_v2_breakdown_noise0.75.csv` (unseen corruption types);
- `benchmark_v3_breakdown_noise0.75.csv`.

Expected overall: ~32.74 dB / 0.874 / 0.137. If the numbers differ noticeably, update the table at the end of `final_submission/generative_denoiser/README.md` before zipping.

```
# 4.2 (optional, ~2 min) the submission package itself on real benchmark images + sample images for the report
#     copy the package folder to the GPU machine first, e.g. scp -r final_submission/generative_denoiser <gpu>:/tmp/
cd /tmp/generative_denoiser
python - "$DATA" <<'EOF'
import sys, json, numpy as np
from pathlib import Path
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from inference import denoise_image
d = Path(sys.argv[1]) / "benchmark" / "v1"; meta = json.load(open(d / "meta.json"))
p, s = [], []
for m in meta:
    x = np.array(Image.open(d / "input" / m["file"]).convert("RGB")); y = np.array(Image.open(d / "target" / m["file"]).convert("RGB"))
    o = denoise_image(x, seed=0); p.append(psnr(y, o, data_range=255)); s.append(ssim(y, o, channel_axis=2, data_range=255))
print("package on all %d benchmark images: PSNR %.2f  SSIM %.4f" % (len(p), np.mean(p), np.mean(s)))
out = Path("/tmp/diversity_samples"); out.mkdir(exist_ok=True)
for i in (0, 150, 333):                       # natural, low-light, medical
    m = meta[i]; x = np.array(Image.open(d / "input" / m["file"]).convert("RGB"))
    samples = [denoise_image(x, seed=k) for k in range(4)]
    std = np.stack(samples).astype(float).std(0).mean(2)
    Image.fromarray(np.concatenate([x] + samples, 1)).save(out / f"{i}_{m['domain']}.png")
    Image.fromarray(np.clip(std * 40, 0, 255).astype(np.uint8)).save(out / f"{i}_std.png")
EOF
```

Expected: PSNR ≈ 32.74, SSIM ≈ 0.874 (the same numbers as 4.1, now from the package). Give `/tmp/diversity_samples/` to the teammate.

## 5. Order today

1. GPU: §4.1 (and optionally §4.2) → numbers and images to the teammate.
2. Laptop: §3.1–3.2 (re-check); update the README results table only if §4 changed the numbers.
3. Laptop: §3.3 zip → submit with the report, target ~20:00, deadline 23:59.
