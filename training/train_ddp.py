"""RLCD fine-tuning for Laya — vendored from the official notebook.

Source: NandhaKishorM/laya, `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`
(Apache-2.0), retrieved 2026-09-22. The training recipe is the maintainer's proven
implementation, and rewriting it would only add ways to be wrong. Only the *data* differs —
`training/build_items.py` produces `train_items.pt` in the shape this script expects.

The single deviation: `EPOCHS`, `LR_ENCODER` and `LR_HEAD` are read from the environment
(`JEV_EPOCHS`, `JEV_LR_ENCODER`, `JEV_LR_HEAD`) so the epoch count can be ablated without editing
this file per experiment. The defaults are the maintainer's originals, so an unset environment
behaves exactly as the vendored version did. Measured motivation: at 4 epochs the training loss is
still halving per epoch (0.686 → 0.515 → 0.174 → 0.081), which is a model that stopped early, not
one that converged.

RLCD (Reinforcement Learning for Calibrated Decisions): the policy reports a distribution;
exploration adds zero-mean Gaussian noise to the logits; the reward is a strictly proper scoring
rule (log + spherical, plus ranked probability score for ordinal questions). Expected reward is
maximised only by reporting honest probabilities — which is the property we need for the
escalation gate to mean anything.

Run on Kaggle with 2xT4:
    torchrun --standalone --nproc_per_node=2 training/train_ddp.py <model_dir> <output_dir>
"""

import os, sys, time, json, random, math, hashlib, platform, importlib.metadata
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from laya.common import build_model, proper_reward, QTYPES


def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def evaluate_validation_items(model, items, tok, device, cfg, batch_size: int = 16):
    """Evaluate a fixed item set without gradients and return per-task proper-score rows."""
    if not items:
        return []
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            batch = collate_train_batch(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            target = batch["target"].to(device)
            for index, item in enumerate(chunk):
                k = int(mask[index].sum().item())
                z = logits[index, :k]
                t = target[index, :k]
                log_probs = torch.log_softmax(z, dim=-1)
                p = torch.softmax(z, dim=-1)
                rows.append(
                    {
                        "task": item.get("task", f"qtype-{int(item.get('qtype', -1))}"),
                        "correct": int(int(p.argmax().item()) == int(t.argmax().item())),
                        "nll": float(-(t * log_probs).sum().item()),
                        "brier": float(((p - t) ** 2).sum().item()),
                    }
                )
    model.train()
    return rows


def summarise_validation(rows):
    """Small local copy of the pure selection rule so the Kaggle script is self-contained."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["task"], []).append(row)
    by_task = {}
    for task, values in sorted(grouped.items()):
        n = len(values)
        by_task[task] = {
            "n": n,
            "accuracy": sum(v["correct"] for v in values) / n,
            "nll": sum(v["nll"] for v in values) / n,
            "brier": sum(v["brier"] for v in values) / n,
        }
    task_scores = [v["nll"] for v in by_task.values()]
    return {
        "by_task": by_task,
        "macro_nll": sum(task_scores) / len(task_scores) if task_scores else None,
    }


def _save_checkpoint(model, tok, output_dir, name, *, epoch, total_epochs, avg_loss, updates, seed):
    ckpt_dir = os.path.join(output_dir, name)
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(ckpt_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(ckpt_dir, "encoder"))
    tok.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))
    with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w") as handle:
        json.dump(
            {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "avg_loss": avg_loss,
                "optimizer_updates": updates,
                "seed": seed,
            },
            handle,
            indent=2,
        )


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _write_run_manifest(output_dir, *, model_dir, items_path, calibration_path, validation_path, cfg, seed, rank, world_size, task_counts):
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "items": {"path": str(items_path), "sha256": _sha256(items_path)},
        "calibration": (
            {"path": str(calibration_path), "sha256": _sha256(calibration_path)}
            if calibration_path
            else {"source": "legacy-fallback", "warning": "calibration was sampled from training items"}
        ),
        "validation": (
            {"path": str(validation_path), "sha256": _sha256(validation_path)}
            if validation_path
            else None
        ),
        "config": {
            "max_len": cfg.get("max_len"),
            "head_max_len": cfg.get("head_max_len"),
            "seed": seed,
            "epochs": int(os.environ.get("JEV_EPOCHS", "4")),
            "lr_encoder": float(os.environ.get("JEV_LR_ENCODER", "2.5e-5")),
            "lr_head": float(os.environ.get("JEV_LR_HEAD", "1.0e-4")),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "packages": {
                name: _package_version(name)
                for name in ("laya", "transformers", "torch", "safetensors", "huggingface-hub", "numpy")
            },
            "base_model": {
                "id": os.environ.get("LAYA_MODEL_ID", "convaiinnovations/laya"),
                "revision": os.environ.get("LAYA_MODEL_REVISION"),
            },
            "rank": rank,
            "world_size": world_size,
        },
        "task_counts": task_counts,
    }
    with open(Path(output_dir) / "run_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model_dir = sys.argv[1]
    output_dir = sys.argv[2]
    items_path = sys.argv[3] if len(sys.argv) > 3 else "/kaggle/working/train_items.pt"
    calibration_path = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("JEV_CALIBRATION_ITEMS")
    validation_path = os.environ.get("JEV_VALIDATION_ITEMS")

    seed = int(os.environ.get("JEV_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = 4096
    # The builder and trainer must use one budget. The legacy script silently changed this after
    # items had already been encoded, which made train_items_meta.json and the saved config disagree.
    cfg["max_len"] = int(os.environ.get("JEV_MAX_LEN", "512"))
    cfg["head_max_len"] = int(os.environ.get("JEV_HEAD_MAX_LEN", "192"))

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))

    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)

    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.head_checkpointing = True
    model.to(device)
    model.train()

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    all_items = torch.load(items_path, weights_only=False)
    task_counts = {}
    for item in all_items:
        task = str(item.get("task", "unknown"))
        task_counts[task] = task_counts.get(task, 0) + 1
    my_items = all_items[rank::world_size]
    if os.environ.get("JEV_REQUIRE_SPLITS", "0") == "1" and not (calibration_path and validation_path):
        raise SystemExit(
            "JEV_REQUIRE_SPLITS=1 requires both JEV_CALIBRATION_ITEMS and JEV_VALIDATION_ITEMS"
        )
    if calibration_path and not os.path.isfile(calibration_path):
        raise SystemExit(f"calibration item file does not exist: {calibration_path}")
    if validation_path and not os.path.isfile(validation_path):
        raise SystemExit(f"validation item file does not exist: {validation_path}")
    calibration_items = (
        torch.load(calibration_path, weights_only=False) if calibration_path else None
    )
    validation_items = (
        torch.load(validation_path, weights_only=False) if validation_path else None
    )
    if rank == 0:
        print(
            f"split inputs: train={items_path} calibration={calibration_path or 'LEGACY-FALLBACK'} "
            f"validation={validation_path or 'none'}"
        )
        if calibration_items is None:
            print(
                "WARNING: calibration is being sampled from training items for backwards "
                "compatibility; this is not a held-out calibration result"
            )

    # The one change to the vendored recipe: these are read from the environment so the epoch
    # count and learning rates can be ablated without editing this file per experiment. The
    # defaults are the maintainer's originals, so an unset environment behaves exactly as before.
    EPOCHS = int(os.environ.get("JEV_EPOCHS", "4"))
    MICRO_BATCH = 8
    GRAD_ACCUM = 4
    GROUP_SIZE = 4
    LR_ENCODER = float(os.environ.get("JEV_LR_ENCODER", "2.5e-5"))
    LR_HEAD = float(os.environ.get("JEV_LR_HEAD", "1.0e-4"))
    SIGMA_START = 0.4
    SIGMA_END = 0.1

    enc_params = [p for n, p in ddp_model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in ddp_model.named_parameters() if "encoder." not in n]

    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": LR_ENCODER},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=0.01,
    )

    updates_per_epoch = max(1, (len(my_items) + MICRO_BATCH * GRAD_ACCUM - 1) // (MICRO_BATCH * GRAD_ACCUM))
    total_updates = updates_per_epoch * EPOCHS
    optimizer_updates = 0
    validation_history = []
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_updates), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    if rank == 0:
        print(
            f"Starting 2xT4 DDP training: {len(all_items)} total items | "
            f"{len(my_items)} per rank | {EPOCHS} epochs"
        )
    t0 = time.time()

    for epoch in range(EPOCHS):
        random.seed(seed + epoch + rank)
        np.random.seed(seed + epoch + rank)
        torch.manual_seed(seed + epoch + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + epoch + rank)
        random.shuffle(my_items)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0

        progress = epoch / max(1, EPOCHS - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress

        for b_idx in range(0, len(my_items), MICRO_BATCH):
            chunk = my_items[b_idx : b_idx + MICRO_BATCH]
            if not chunk:
                continue

            batch = collate_train_batch(chunk, tok.pad_token_id)

            with torch.autocast("cuda", dtype=torch.float16):
                logits, act = ddp_model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )

            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            with torch.no_grad():
                r = proper_reward(
                    q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0
                )
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(
                target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
            ).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum_step += 1

            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # GradScaler may skip an optimizer step on non-finite gradients. The scheduler
                # must not advance for a step that did not happen (the v7 log showed this warning).
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                    optimizer_updates += 1
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1

            if rank == 0 and (n_batches % 50) == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch+1}/{EPOCHS} | Step {n_batches} | "
                    f"Loss: {loss.item()*GRAD_ACCUM:.4f} | Reward: {r.mean().item():.3f} | "
                    f"LR: {cur_lr:.2e}"
                )

        if rank == 0:
            print(
                f"=== Epoch {epoch+1}/{EPOCHS} Completed in {time.time()-t0:.1f}s | "
                f"Avg Loss: {epoch_loss/max(1, n_batches):.4f} ==="
            )

        dist.barrier()

        if validation_items is not None:
            # Only rank 0 runs the deterministic validation pass; the other ranks wait at the
            # barrier. This avoids invoking the DDP wrapper outside a gradient-synchronised step.
            if rank == 0:
                val_rows = evaluate_validation_items(model, validation_items, tok, device, cfg)
                val_summary = summarise_validation(val_rows)
                validation_history.append({"epoch": epoch + 1, **val_summary})
                with open(os.path.join(output_dir, "validation_history.json"), "w") as handle:
                    json.dump(validation_history, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                eligible = [row for row in validation_history if row.get("macro_nll") is not None]
                best = min(eligible, key=lambda row: row["macro_nll"]) if eligible else None
                if best is not None and best["epoch"] == epoch + 1:
                    _save_checkpoint(
                        model,
                        tok,
                        output_dir,
                        "checkpoint_best",
                        epoch=epoch + 1,
                        total_epochs=EPOCHS,
                        avg_loss=epoch_loss / max(1, n_batches),
                        updates=optimizer_updates,
                        seed=seed,
                    )
                print(f"  validation: {val_summary}")
            dist.barrier()

        if rank == 0:
            _save_checkpoint(
                model,
                tok,
                output_dir,
                "checkpoint_latest",
                epoch=epoch + 1,
                total_epochs=EPOCHS,
                avg_loss=epoch_loss / max(1, n_batches),
                updates=optimizer_updates,
                seed=seed,
            )
            print(f"  Saved rolling checkpoint (epoch {epoch+1}/{EPOCHS}) to checkpoint_latest")

    dist.barrier()

    # Post-training temperature calibration (this is what makes the confidence usable).
    if rank == 0:
        print("\nFitting post-training calibration temperatures...")
        del optimizer, scaler, scheduler
        torch.cuda.empty_cache()
        model.eval()
        calib_items = calibration_items if calibration_items is not None else all_items[::15][:400]
        if calibration_items is None:
            print(
                "WARNING: fitting temperature on legacy training items; pass a separate "
                "calibration file for an honest calibration result"
            )
        calib_preds = []
        with torch.no_grad():
            for c_idx in range(0, len(calib_items), 16):
                c_chunk = calib_items[c_idx : c_idx + 16]
                cb = collate_train_batch(c_chunk, tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.float16):
                    l_sub, _ = model(
                        cb["input_ids"].to(device),
                        cb["attention_mask"].to(device),
                        cb["marker_pos"].to(device),
                        cb["marker_mask"].to(device),
                        cb["qtype"].to(device),
                    )
                l_np = l_sub.float().cpu().numpy()
                for r, it in enumerate(c_chunk):
                    k = len(it["markers"])
                    calib_preds.append((it["qtype"], l_np[r, :k], it["target"]))

        fitted_temps = [1.2, 1.2, 1.2]
        try:
            for qt in range(3):
                sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
                if sel:
                    fitted_temps[qt] = fit_one_temp(sel)
            print(
                "Fitted calibration temperatures (choice, score, noul):",
                [round(t, 3) for t in fitted_temps],
            )
        except Exception as e:
            print("Temperature fitting fallback:", e)

        os.makedirs(output_dir, exist_ok=True)
        sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
        save_file(sd, os.path.join(output_dir, "model.safetensors"))
        model.encoder.config.save_pretrained(os.path.join(output_dir, "encoder"))
        tok.save_pretrained(os.path.join(output_dir, "tokenizer"))

        cfg["fine_tuned"] = True
        cfg["model_name"] = "laya-dealership-routing"
        cfg["temperature"] = fitted_temps
        # `temperature_by_options` is inherited from the base checkpoint and *takes precedence* over
        # the vector fitted just above (laya_mlx/agent.py: `temperature_by_options.get(bucket, ...)`,
        # the fitted value only as fallback). Every bucket our questions occupy - noul:2, choice:2,
        # choice:3-5, choice:6-10 - is present in that inherited map, so the fitted temperatures were
        # never applied to a single question. The step the comment above calls "what makes the
        # confidence usable" was a no-op for this task. Measured on the v5 checkpoint: removing the
        # map moves a noul probability from 1.000 to 0.996, so this is a correctness fix rather than
        # a rescue - the trained logits are extreme in their own right.
        cfg.pop("temperature_by_options", None)
        with open(os.path.join(output_dir, "rl_agent_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Model successfully saved to {output_dir}!")
        _write_run_manifest(
            output_dir,
            model_dir=model_dir,
            items_path=items_path,
            calibration_path=calibration_path,
            validation_path=validation_path,
            cfg=cfg,
            seed=seed,
            rank=rank,
            world_size=world_size,
            task_counts=task_counts,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
