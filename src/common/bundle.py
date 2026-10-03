"""
Shared ModelBundle for the round-2 architectures (a05 onward).

It implements the Trainer interface documented in src/common/trainer.py once,
so each architecture's model.py only defines its network. What it adds over
the round-1 bundles:

  - per-STEP warmup + cosine learning rate (round 1 stepped per epoch);
  - mixed precision on CUDA, which roughly halves memory and time. If the
    loss turns non-finite repeatedly it switches itself off and carries on
    in full precision rather than wasting the run;
  - an EMA copy of the weights, which is what gets validated, checkpointed
    as the inference model and exported;
  - the composite loss (pixel + frequency + SSIM + DISTS).

Subclasses override `forward_train` when training is not a plain
corrupted -> clean regression (a08 adds an auxiliary loss, a10 trains a flow).
"""
import copy
import math

import torch

from .layers import count_parameters
from .losses import CompositeLoss


def _grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):            # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


class RestorationBundle:
    uses_labels = False          # set True to receive the degradation label vector

    def __init__(self, cfg, device, model, name="model"):
        t = cfg["train"]
        self.cfg, self.device, self.name = cfg, device, name
        self.on_cuda = device.type == "cuda"

        self.model = model.to(device)
        if self.on_cuda:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.ema = copy.deepcopy(self.model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.ema_decay = t.get("ema_decay", 0.999)

        self.criterion = CompositeLoss(t.get("loss")).to(device)

        self.base_lr = t.get("lr", 4e-4)
        self.lr_min = t.get("lr_min", 1e-6)
        self.warmup_steps = t.get("warmup_steps", 1000)
        self.total_steps = t["epochs"] * cfg["data"].get("steps_per_epoch", 2500)
        self.grad_clip = t.get("grad_clip", 1.0)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.base_lr, betas=(0.9, 0.99),
            weight_decay=t.get("weight_decay", 1e-4))

        self.use_amp = bool(t.get("amp", True)) and self.on_cuda
        self.scaler = _grad_scaler(self.use_amp)
        self.step = 0
        self.bad_steps = 0
        print(f"[{name}] parameters: {count_parameters(self.model):,}  "
              f"| total steps: {self.total_steps:,}  | amp: {self.use_amp}")

    # ---- overridable ---- #
    def forward_train(self, corrupted, clean, labels):
        """Returns (prediction, extra_loss_tensor_or_None, extra_log_dict)."""
        return self.model(corrupted), None, {}

    def inference(self, model, corrupted):
        return model(corrupted)

    # ---- Trainer interface ---- #
    def _lr_at(self, step):
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        span = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / span)
        return self.lr_min + 0.5 * (self.base_lr - self.lr_min) * (1 + math.cos(math.pi * progress))

    def train_step(self, corrupted, clean, labels=None):
        self.model.train()
        for group in self.optimizer.param_groups:
            group["lr"] = self._lr_at(self.step)
        if self.on_cuda:
            corrupted = corrupted.contiguous(memory_format=torch.channels_last)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
            pred, extra, logs = self.forward_train(corrupted, clean, labels)
        loss, parts = self.criterion(pred, clean, self.step / self.total_steps)
        if extra is not None:
            loss = loss + extra.float()

        if not torch.isfinite(loss):
            return self._skip_step()

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
        if not torch.isfinite(grad_norm):
            self.scaler.update()
            return self._skip_step()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.step += 1
        self._update_ema()
        return {"loss": loss.item(), **parts, **logs}

    def _skip_step(self):
        """A non-finite loss or gradient: make no update, so the weights can never
        be poisoned. If it keeps happening under mixed precision, half-precision
        arithmetic is the cause, so fall back to float32 for the rest of the run."""
        self.optimizer.zero_grad(set_to_none=True)
        self.bad_steps += 1
        if self.use_amp and self.bad_steps >= 20:
            print(f"[{self.name}] WARNING: {self.bad_steps} non-finite steps; "
                  f"disabling mixed precision for the rest of the run.", flush=True)
            self.use_amp = False
            self.scaler = _grad_scaler(False)
        self.step += 1
        return {"loss": float("nan")}

    @torch.no_grad()
    def _update_ema(self):
        decay = min(self.ema_decay, (1 + self.step) / (10 + self.step))   # fast early, slow later
        for e, p in zip(self.ema.parameters(), self.model.parameters()):
            e.lerp_(p.detach(), 1.0 - decay)

    @torch.no_grad()
    def eval_step(self, corrupted, clean):
        pred = self.inference(self.ema, corrupted).float().clamp(0, 1)
        return pred, {"loss": self.criterion.pix(pred, clean).item()}

    def get_inference_model(self):
        return self.ema

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self):
        return {"model": self.model.state_dict(), "ema": self.ema.state_dict(),
                "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(),
                "step": self.step}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.ema.load_state_dict(state.get("ema", state["model"]))
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state and self.use_amp:
            self.scaler.load_state_dict(state["scaler"])
        self.step = state.get("step", 0)
