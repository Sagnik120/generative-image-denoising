"""
Generic, architecture-agnostic training loop.

Every architecture module (nafnet_unet, restormer_lite, pix2pix_gan,
tiny_ddpm_sr3, ...) exposes a `build(cfg, device)` function returning a
`ModelBundle` with this interface:

    bundle.train_step(corrupted, clean) -> dict of scalar losses (floats),
                                             must include "loss" (the primary
                                             scalar used for the loss curve).
    bundle.eval_step(corrupted, clean)  -> (prediction, dict of scalar losses)
    bundle.get_inference_model()        -> nn.Module used for FLOPs counting
                                             and for the final saved weights.
    bundle.state_dict()                 -> dict to checkpoint (models + optimizers)
    bundle.load_state_dict(state)       -> restore from checkpoint

This lets the Trainer below stay 100% identical across every architecture
you add -- you only ever write a new architecture's model.py, never a new
training loop.
"""
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .utils import AverageMeter, CSVLogger, save_checkpoint, save_json, ensure_dirs
from .metrics import evaluate_batch, compute_flops_and_params
from .visualize import plot_loss_curves, plot_metric_curves, save_comparison_grid, save_error_heatmap


class Trainer:
    def __init__(self, cfg, bundle, train_loader, val_loader, results_dir, arch_name, device):
        self.cfg = cfg
        self.bundle = bundle
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.arch_name = arch_name

        self.results_dir = Path(results_dir)
        self.dirs = {
            "checkpoints": self.results_dir / "checkpoints",
            "loss_curves": self.results_dir / "loss_curves",
            "metrics": self.results_dir / "metrics",
            "visualizations": self.results_dir / "visualizations",
            "logs": self.results_dir / "logs",
        }
        ensure_dirs(*self.dirs.values())

        self.csv_path = self.dirs["logs"] / "training_log.csv"
        self.logger = CSVLogger(
            self.csv_path,
            fieldnames=["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim",
                        "val_dists", "gen_loss", "disc_loss", "lr", "epoch_time_sec"],
        )

        self.best_score = -float("inf")
        self.best_ckpt_path = self.dirs["checkpoints"] / "best_model.pt"
        self.last_ckpt_path = self.dirs["checkpoints"] / "last_model.pt"

    def _composite_score(self, psnr, ssim, dists):
        # Higher is better composite: reward PSNR/SSIM, penalize DISTS.
        # (Used only for local "best checkpoint" selection -- the actual
        # competition ranks each axis independently; see docs/ for detail.)
        dists_term = 0.0 if dists != dists else (1.0 - dists)  # NaN-safe
        return psnr / 40.0 + ssim + dists_term

    def train(self):
        cfg = self.cfg
        epochs = cfg["train"]["epochs"]
        vis_every = cfg["train"].get("visualize_every", 1)
        save_every = cfg["train"].get("checkpoint_every", 1)

        print(f"[trainer:{self.arch_name}] Starting training for {epochs} epochs "
              f"on device={self.device}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self._train_one_epoch(epoch)
            val_metrics = self._validate(epoch, save_visuals=(epoch % vis_every == 0))
            elapsed = time.time() - t0

            row = {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss", float("nan")),
                "val_loss": val_metrics.get("loss", float("nan")),
                "val_psnr": val_metrics.get("psnr", float("nan")),
                "val_ssim": val_metrics.get("ssim", float("nan")),
                "val_dists": val_metrics.get("dists", float("nan")),
                "gen_loss": train_metrics.get("gen_loss", ""),
                "disc_loss": train_metrics.get("disc_loss", ""),
                "lr": self.bundle.get_lr() if hasattr(self.bundle, "get_lr") else "",
                "epoch_time_sec": round(elapsed, 2),
            }
            self.logger.log(row)

            print(f"[epoch {epoch}/{epochs}] "
                  f"train_loss={row['train_loss']:.4f}  val_loss={row['val_loss']:.4f}  "
                  f"PSNR={row['val_psnr']:.2f}  SSIM={row['val_ssim']:.4f}  "
                  f"DISTS={row['val_dists']:.4f}  ({elapsed:.1f}s)")

            score = self._composite_score(row["val_psnr"], row["val_ssim"], row["val_dists"])
            if score > self.best_score:
                self.best_score = score
                save_checkpoint(self.bundle.state_dict(), self.best_ckpt_path)
                print(f"  -> new best model saved (composite score={score:.4f})")

            if epoch % save_every == 0:
                save_checkpoint(self.bundle.state_dict(), self.last_ckpt_path)

            if hasattr(self.bundle, "step_scheduler"):
                self.bundle.step_scheduler()

        # ---- Post-training artifacts ----
        plot_loss_curves(self.csv_path, self.dirs["loss_curves"], self.arch_name)
        plot_metric_curves(self.csv_path, self.dirs["metrics"], self.arch_name)
        self._final_report()
        print(f"[trainer:{self.arch_name}] Training complete. "
              f"Best checkpoint: {self.best_ckpt_path}")

    def _train_one_epoch(self, epoch):
        loss_meter = AverageMeter()
        gen_meter = AverageMeter()
        disc_meter = AverageMeter()
        has_gan = False

        pbar = tqdm(self.train_loader, desc=f"train epoch {epoch}", leave=False)
        for corrupted, clean in pbar:
            corrupted = corrupted.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)

            losses = self.bundle.train_step(corrupted, clean)
            loss_meter.update(losses["loss"], n=corrupted.size(0))
            if "gen_loss" in losses:
                has_gan = True
                gen_meter.update(losses["gen_loss"], n=corrupted.size(0))
                disc_meter.update(losses["disc_loss"], n=corrupted.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        out = {"loss": loss_meter.avg}
        if has_gan:
            out["gen_loss"] = gen_meter.avg
            out["disc_loss"] = disc_meter.avg
        return out

    @torch.no_grad()
    def _validate(self, epoch, save_visuals=False):
        loss_meter = AverageMeter()
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        dists_meter = AverageMeter()

        first_batch = None
        for i, (corrupted, clean) in enumerate(tqdm(self.val_loader, desc="validate", leave=False)):
            corrupted = corrupted.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)

            pred, losses = self.bundle.eval_step(corrupted, clean)
            loss_meter.update(losses["loss"], n=corrupted.size(0))

            m = evaluate_batch(pred, clean, self.device)
            psnr_meter.update(m["psnr"], n=corrupted.size(0))
            ssim_meter.update(m["ssim"], n=corrupted.size(0))
            if m["dists"] == m["dists"]:  # skip NaN
                dists_meter.update(m["dists"], n=corrupted.size(0))

            if i == 0:
                first_batch = (corrupted.cpu(), pred.cpu(), clean.cpu())

        if save_visuals and first_batch is not None:
            corrupted_cpu, pred_cpu, clean_cpu = first_batch
            save_comparison_grid(
                corrupted_cpu, pred_cpu, clean_cpu,
                out_path=self.dirs["visualizations"] / f"epoch_{epoch:04d}_comparison.png",
                epoch=epoch, arch_name=self.arch_name,
            )
            save_error_heatmap(
                pred_cpu, clean_cpu,
                out_path=self.dirs["visualizations"] / f"epoch_{epoch:04d}_error_heatmap.png",
            )

        return {
            "loss": loss_meter.avg,
            "psnr": psnr_meter.avg,
            "ssim": ssim_meter.avg,
            "dists": dists_meter.avg if dists_meter.count > 0 else float("nan"),
        }

    def _final_report(self):
        model = self.bundle.get_inference_model()
        flop_info = compute_flops_and_params(model, input_size=(1, 3, 256, 256),
                                              device=self.device)

        # Some architectures (e.g. tiny_ddpm_sr3) run multiple forward passes per
        # inference call (T diffusion steps). If the model config declares
        # `inference_steps`, report the TRUE total FLOPs too, so nothing is
        # silently underreported relative to what the competition will measure.
        inference_steps = self.cfg.get("model", {}).get("timesteps", 1)
        true_total_gflops = flop_info["gflops"] * inference_steps

        report = {
            "architecture": self.arch_name,
            "best_composite_score": self.best_score,
            "flops_per_forward_call": flop_info["flops"],
            "gflops_per_forward_call": flop_info["gflops"],
            "inference_forward_passes": inference_steps,
            "true_total_gflops_at_inference": true_total_gflops,
            "params": flop_info["params"],
            "config": self.cfg,
        }
        save_json(report, self.dirs["metrics"] / "final_report.json")
        print(f"[trainer:{self.arch_name}] FLOPs/call: {flop_info['gflops']:.3f} GFLOPs "
              f"(x{inference_steps} steps = {true_total_gflops:.3f} GFLOPs true total), "
              f"Params: {flop_info['params']:,}")
