"""
Generic, architecture-agnostic training loop.

Every architecture module (a01_nafnet_unet, a02_restormer_lite, a03_pix2pix_gan,
a04_tiny_ddpm_sr3, ...) exposes a `build(cfg, device)` function returning a
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

Optional extras a bundle may provide:
    bundle.uses_labels = True  -> train_step also receives the degradation
                                  label vector as a third argument.
    bundle.step_scheduler()    -> called once per epoch (round-1 bundles; the
                                  round-2 bundle schedules per step itself).

An "epoch" here is `data.steps_per_epoch` training steps followed by one
pass over the fixed benchmark. Every epoch ends with an atomic checkpoint
that also stores the epoch number and best score, so an interrupted run
continues exactly where it stopped (scripts/train.py --resume auto).
"""
import shutil
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .utils import AverageMeter, CSVLogger, save_checkpoint, save_json, ensure_dirs
from .metrics import evaluate_batch_per_image, summarize_benchmark, compute_flops_and_params
from .visualize import plot_loss_curves, plot_metric_curves, save_comparison_grid, save_error_heatmap


class Trainer:
    def __init__(self, cfg, bundle, train_loader, val_loader, results_dir, arch_name, device,
                 dry_run_steps: int = 0):
        self.cfg = cfg
        self.bundle = bundle
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.arch_name = arch_name
        self.dry_run_steps = dry_run_steps
        self.val_meta = getattr(val_loader.dataset, "meta", None)
        self.interactive = sys.stderr.isatty()     # progress bars only on a real terminal

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
        self.fieldnames = ["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim",
                           "val_dists", "gen_loss", "disc_loss", "lr", "epoch_time_sec"]

        self.start_epoch = 1
        self.best_score = -float("inf")
        self.best_ckpt_path = self.dirs["checkpoints"] / "best_model.pt"
        self.last_ckpt_path = self.dirs["checkpoints"] / "last_model.pt"

    def resume(self, trainer_state):
        """Continue after the epoch stored in a checkpoint's "trainer" entry."""
        if trainer_state:
            self.start_epoch = trainer_state["epoch"] + 1
            self.best_score = trainer_state["best_score"]
            print(f"[trainer:{self.arch_name}] Resuming at epoch {self.start_epoch} "
                  f"(best score so far {self.best_score:.4f}).")

    def _composite_score(self, psnr, ssim, dists):
        # Higher is better composite: reward PSNR/SSIM, penalize DISTS.
        # (Used only for local "best checkpoint" selection -- the actual
        # competition ranks each axis independently; see docs/ for detail.)
        dists_term = 0.0 if dists != dists else (1.0 - dists)  # NaN-safe
        return psnr / 40.0 + ssim + dists_term

    def _save(self, path, epoch):
        state = self.bundle.state_dict()
        state["trainer"] = {"epoch": epoch, "best_score": self.best_score}
        save_checkpoint(state, path)

    def train(self):
        cfg = self.cfg
        epochs = cfg["train"]["epochs"]
        vis_every = cfg["train"].get("visualize_every", 1)
        save_every = cfg["train"].get("checkpoint_every", 1)

        if self.start_epoch == 1 and self.csv_path.exists():
            self.csv_path.replace(self.csv_path.with_suffix(".previous.csv"))   # fresh run
        self.logger = CSVLogger(self.csv_path, fieldnames=self.fieldnames)

        print(f"[trainer:{self.arch_name}] Training epochs {self.start_epoch}..{epochs} "
              f"({len(self.train_loader)} steps each) on device={self.device}")

        for epoch in range(self.start_epoch, epochs + 1):
            t0 = time.time()
            train_metrics = self._train_one_epoch(epoch, epochs)
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
                  f"DISTS={row['val_dists']:.4f}  ({elapsed:.1f}s)", flush=True)

            score = self._composite_score(row["val_psnr"], row["val_ssim"], row["val_dists"])
            if score > self.best_score:
                self.best_score = score
                self._save(self.best_ckpt_path, epoch)
                last = self.dirs["metrics"] / "benchmark_breakdown_last.json"
                if last.exists():
                    shutil.copyfile(last, self.dirs["metrics"] / "benchmark_breakdown_best.json")
                print(f"  -> new best model saved (composite score={score:.4f})")

            if epoch % save_every == 0 or epoch == epochs:
                self._save(self.last_ckpt_path, epoch)

            if hasattr(self.bundle, "step_scheduler"):
                self.bundle.step_scheduler()

        # ---- Post-training artifacts ----
        if self.csv_path.exists():
            plot_loss_curves(self.csv_path, self.dirs["loss_curves"], self.arch_name)
            plot_metric_curves(self.csv_path, self.dirs["metrics"], self.arch_name)
        self._final_report()
        print(f"[trainer:{self.arch_name}] Training complete. "
              f"Best checkpoint: {self.best_ckpt_path}", flush=True)

    def _train_one_epoch(self, epoch, epochs):
        meters = {}
        has_gan = False
        n_steps = len(self.train_loader)
        log_every = self.cfg["train"].get("log_every", 250)
        uses_labels = getattr(self.bundle, "uses_labels", False)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        loader = tqdm(self.train_loader, desc=f"train epoch {epoch}", leave=False) \
            if self.interactive else self.train_loader
        for step, batch in enumerate(loader, 1):
            corrupted = batch[0].to(self.device, non_blocking=True)
            clean = batch[1].to(self.device, non_blocking=True)

            if uses_labels and len(batch) > 2:
                losses = self.bundle.train_step(corrupted, clean, batch[2].to(self.device))
            else:
                losses = self.bundle.train_step(corrupted, clean)
            for key, value in losses.items():
                if value == value:                      # skip NaN (a skipped step)
                    meters.setdefault(key, AverageMeter()).update(value, n=corrupted.size(0))
            has_gan = has_gan or "gen_loss" in losses

            if step % log_every == 0 or step == n_steps or step == self.dry_run_steps:
                rate = step / (time.time() - t0)
                mem = f"  gpu_mem={torch.cuda.max_memory_allocated() / 1024 ** 3:.1f}GB" \
                    if self.device.type == "cuda" else ""
                parts = "  ".join(f"{k}={m.avg:.4f}" for k, m in meters.items())
                print(f"  [epoch {epoch}/{epochs} step {step}/{n_steps}] {parts}  "
                      f"{rate:.2f} it/s  epoch_eta={(n_steps - step) / rate / 60:.1f}min{mem}",
                      flush=True)
            if self.dry_run_steps and step >= self.dry_run_steps:
                break

        out = {"loss": meters["loss"].avg if "loss" in meters else float("nan")}
        if has_gan:
            out["gen_loss"] = meters["gen_loss"].avg
            out["disc_loss"] = meters["disc_loss"].avg
        return out

    @torch.no_grad()
    def _validate(self, epoch, save_visuals=False):
        loss_meter = AverageMeter()
        records = []

        first_batch = None
        loader = tqdm(self.val_loader, desc="validate", leave=False) \
            if self.interactive else self.val_loader
        for i, batch in enumerate(loader):
            corrupted = batch[0].to(self.device, non_blocking=True)
            clean = batch[1].to(self.device, non_blocking=True)

            pred, losses = self.bundle.eval_step(corrupted, clean)
            loss_meter.update(losses["loss"], n=corrupted.size(0))

            m = evaluate_batch_per_image(pred, clean, self.device)
            records += [{"psnr": p, "ssim": s, "dists": d}
                        for p, s, d in zip(m["psnr"], m["ssim"], m["dists"])]

            if i == 0:
                first_batch = (corrupted.cpu(), pred.cpu(), clean.cpu())
            if self.dry_run_steps and i >= 1:
                break

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

        mean = lambda key: (lambda v: sum(v) / len(v) if v else float("nan"))(
            [r[key] for r in records if r[key] == r[key]])
        if self.val_meta is not None:
            summary = summarize_benchmark(records, self.val_meta)
            summary["epoch"] = epoch
            save_json(summary, self.dirs["metrics"] / "benchmark_breakdown_last.json")
            for field in ("by_group", "by_domain"):
                print("  " + field[3:] + ": " + "  ".join(
                    f"{k}={v['psnr']:.2f}/{v['ssim']:.3f}/{v['dists']:.3f}"
                    for k, v in summary[field].items()) + "   (PSNR/SSIM/DISTS)")

        return {"loss": loss_meter.avg, "psnr": mean("psnr"), "ssim": mean("ssim"),
                "dists": mean("dists")}

    def _final_report(self):
        model = self.bundle.get_inference_model()
        flop_info = compute_flops_and_params(model, input_size=(1, 3, 256, 256),
                                              device=self.device)

        # Some architectures (e.g. a04_tiny_ddpm_sr3) run multiple forward passes per
        # inference call (T diffusion steps). If the model config declares
        # `inference_steps`, report the TRUE total FLOPs too, so nothing is
        # silently underreported relative to what the competition will measure.
        # (a10's sampler runs all of its passes inside one forward call, so its
        # count is already the total and it declares no `timesteps`.)
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
