"""Neural surrogate for GPTZero: DistilBERT fine-tuned with perturbation-contrastive loss.

Architecture:
  DistilBERT-base-uncased -> CLS pooling -> Linear(768,256) -> ReLU -> Dropout -> Linear(256,1) -> Sigmoid

Training objectives:
  L_prob:       MSE on predicted probability vs GPTZero average_generated_prob
  L_class:      Cross-entropy on predicted class (ai/human/mixed) — auxiliary
  L_transition: Margin ranking loss on perturbation pairs — directly targets transition agreement

Inference:
  text -> tokenize -> forward pass -> predicted probability in [0, 1]
  No GPTZero API needed at inference time.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CLASS_LABELS = ["human", "mixed", "ai"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_LABELS)}
DEFAULT_MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LEN = 512


class GPTZeroSurrogate:
    """Inference wrapper around a trained surrogate checkpoint."""

    def __init__(self, model_dir: str, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir)

        meta_path = self.model_dir / "surrogate_meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = _load_model(str(self.model_dir), self.device)
        self.model.eval()

    def predict_prob(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Predict GPTZero-style average_generated_prob for a batch of texts."""
        import torch

        all_probs: List[float] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = list(texts[i : i + batch_size])
            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
                logits = out.logits
                probs = torch.sigmoid(logits[:, 0]).cpu().numpy()
                all_probs.extend(probs.tolist())
        return np.array(all_probs, dtype=float)

    def predict_class(self, texts: Sequence[str], batch_size: int = 32) -> List[str]:
        """Predict class labels. Derived from probability thresholds."""
        probs = self.predict_prob(texts, batch_size)
        labels = []
        for p in probs:
            if p >= 0.65:
                labels.append("ai")
            elif p <= 0.35:
                labels.append("human")
            else:
                labels.append("mixed")
        return labels


import torch
import torch.nn as nn


def _load_model(model_dir: str, device: str) -> Any:
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_dir)
    backbone = AutoModel.from_pretrained(model_dir, config=config)
    model = _SurrogateHead(backbone, config.hidden_size)

    weights_path = Path(model_dir) / "surrogate_head.pt"
    if weights_path.exists():
        head_state = torch.load(str(weights_path), map_location=device, weights_only=True)
        if "prob_head" in head_state:
            model.prob_head.load_state_dict(head_state["prob_head"])
            model.class_head.load_state_dict(head_state["class_head"])
        else:
            model.load_state_dict(head_state, strict=False)

    return model.to(device)


class _SurrogateHead(nn.Module):
    """DistilBERT backbone + regression head for probability prediction."""

    def __init__(self, backbone: nn.Module, hidden_size: int, num_classes: int = 3):
        super().__init__()
        self.backbone = backbone
        self.prob_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        self.class_head = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        prob_logit = self.prob_head(cls_hidden)
        class_logits = self.class_head(cls_hidden)

        class _Out:
            pass

        result = _Out()
        result.logits = prob_logit
        result.class_logits = class_logits
        result.cls_hidden = cls_hidden
        return result


class _PairDataset(torch.utils.data.Dataset):
    """Dataset that yields (base, variant) pairs for contrastive training."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: Any,
        max_len: int = MAX_SEQ_LEN,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.bases: List[Dict] = []
        self.variants: List[Dict] = []
        self.singletons: List[Dict] = []

        groups: Dict[str, List[Dict]] = {}
        for rec in records:
            bid = rec.get("base_sample_id", rec.get("sample_id", ""))
            groups.setdefault(bid, []).append(rec)

        for bid, group in groups.items():
            group.sort(key=lambda r: int(r.get("variant_order", 0)))
            base_rec = group[0]
            self.singletons.append(base_rec)
            for var_rec in group[1:]:
                self.singletons.append(var_rec)
                if base_rec.get("gptzero_prob") is not None and var_rec.get("gptzero_prob") is not None:
                    self.bases.append(base_rec)
                    self.variants.append(var_rec)

        self._pair_count = len(self.bases)
        self._singleton_count = len(self.singletons)

    def __len__(self) -> int:
        return self._singleton_count + self._pair_count

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < self._singleton_count:
            rec = self.singletons[idx]
            enc = self.tokenizer(
                rec["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            prob = float(rec.get("gptzero_prob", 0.0) or 0.0)
            cls = CLASS_TO_IDX.get(str(rec.get("gptzero_class", "human")), 0)
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "prob_target": torch.tensor(prob, dtype=torch.float),
                "class_target": torch.tensor(cls, dtype=torch.long),
                "is_pair": torch.tensor(0, dtype=torch.long),
                "pair_input_ids": torch.zeros(self.max_len, dtype=torch.long),
                "pair_attention_mask": torch.zeros(self.max_len, dtype=torch.long),
                "pair_prob_target": torch.tensor(0.0, dtype=torch.float),
            }
        else:
            pair_idx = idx - self._singleton_count
            base_rec = self.bases[pair_idx]
            var_rec = self.variants[pair_idx]

            base_enc = self.tokenizer(
                base_rec["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            var_enc = self.tokenizer(
                var_rec["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            base_prob = float(base_rec.get("gptzero_prob", 0.0) or 0.0)
            var_prob = float(var_rec.get("gptzero_prob", 0.0) or 0.0)
            cls = CLASS_TO_IDX.get(str(base_rec.get("gptzero_class", "human")), 0)

            return {
                "input_ids": base_enc["input_ids"].squeeze(0),
                "attention_mask": base_enc["attention_mask"].squeeze(0),
                "prob_target": torch.tensor(base_prob, dtype=torch.float),
                "class_target": torch.tensor(cls, dtype=torch.long),
                "is_pair": torch.tensor(1, dtype=torch.long),
                "pair_input_ids": var_enc["input_ids"].squeeze(0),
                "pair_attention_mask": var_enc["attention_mask"].squeeze(0),
                "pair_prob_target": torch.tensor(var_prob, dtype=torch.float),
            }


def _compute_transition_loss(
    base_probs: torch.Tensor,
    var_probs: torch.Tensor,
    base_targets: torch.Tensor,
    var_targets: torch.Tensor,
    margin: float = 0.02,
    tau: float = 0.05,
) -> torch.Tensor:
    """Combined sign-BCE + delta-MSE loss for transition direction matching.

    Two components:
    1. BCE on sign prediction: treats transition direction as binary classification.
       Uses temperature scaling so gradient flows even when deltas are tiny.
    2. MSE on delta magnitude: directly regresses the probability change.
    """
    delta_true = var_targets - base_targets
    delta_pred = var_probs - base_probs

    near_zero = delta_true.abs() < 0.003
    valid = ~near_zero
    if valid.sum() == 0:
        return torch.tensor(0.0, device=base_probs.device)

    dt = delta_true[valid]
    dp = delta_pred[valid]

    sign_target = (torch.sign(dt) + 1.0) / 2.0
    sign_pred = torch.sigmoid(dp / max(tau, 1e-6))
    sign_loss = nn.functional.binary_cross_entropy(sign_pred, sign_target)

    delta_mse = nn.functional.mse_loss(dp, dt)

    return sign_loss + 0.5 * delta_mse


class _PairOnlyDataset(torch.utils.data.Dataset):
    """Dataset that yields ONLY (base, variant) pairs for transition-focused training."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: Any,
        max_len: int = MAX_SEQ_LEN,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bases: List[Dict] = []
        self.variants: List[Dict] = []

        groups: Dict[str, List[Dict]] = {}
        for rec in records:
            bid = rec.get("base_sample_id", rec.get("sample_id", ""))
            groups.setdefault(bid, []).append(rec)

        for bid, group in groups.items():
            group.sort(key=lambda r: int(r.get("variant_order", 0)))
            base_rec = group[0]
            for var_rec in group[1:]:
                if base_rec.get("gptzero_prob") is not None and var_rec.get("gptzero_prob") is not None:
                    delta = abs(float(base_rec["gptzero_prob"]) - float(var_rec["gptzero_prob"]))
                    if delta >= 0.003:
                        self.bases.append(base_rec)
                        self.variants.append(var_rec)

    def __len__(self) -> int:
        return len(self.bases)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_rec = self.bases[idx]
        var_rec = self.variants[idx]

        base_enc = self.tokenizer(
            base_rec["text"], padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        var_enc = self.tokenizer(
            var_rec["text"], padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        base_prob = float(base_rec.get("gptzero_prob", 0.0) or 0.0)
        var_prob = float(var_rec.get("gptzero_prob", 0.0) or 0.0)
        cls = CLASS_TO_IDX.get(str(base_rec.get("gptzero_class", "human")), 0)

        return {
            "input_ids": base_enc["input_ids"].squeeze(0),
            "attention_mask": base_enc["attention_mask"].squeeze(0),
            "prob_target": torch.tensor(base_prob, dtype=torch.float),
            "class_target": torch.tensor(cls, dtype=torch.long),
            "is_pair": torch.tensor(1, dtype=torch.long),
            "pair_input_ids": var_enc["input_ids"].squeeze(0),
            "pair_attention_mask": var_enc["attention_mask"].squeeze(0),
            "pair_prob_target": torch.tensor(var_prob, dtype=torch.float),
        }


def _load_training_data(data_path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    p = Path(data_path)
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("has_error"):
                continue
            if rec.get("gptzero_prob") is None:
                continue
            records.append(rec)
    return records


def train(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train the neural surrogate with two-phase schedule.

    Phase 1 (prob-focus): mixed batches, low transition weight - learn good probabilities.
    Phase 2 (transition-focus): pair-only batches, high transition weight - learn perturbation sensitivity.
    """
    from transformers import AutoModel, AutoConfig, AutoTokenizer

    data_path = str(config.get("data_path", "/artifacts/gptzero-audit/combined_training_data.jsonl"))
    output_dir = str(config.get("output_dir", "/artifacts/gptzero-audit/neural_surrogate"))
    model_name = str(config.get("model_name", DEFAULT_MODEL_NAME))
    max_len = int(config.get("max_seq_len", MAX_SEQ_LEN))
    batch_size = int(config.get("batch_size", 32))
    lr = float(config.get("learning_rate", 2e-5))
    phase1_epochs = int(config.get("phase1_epochs", 5))
    phase2_epochs = int(config.get("phase2_epochs", 10))
    lambda_cls = float(config.get("lambda_class", 0.3))
    lambda_trans_p1 = float(config.get("lambda_transition_phase1", 0.5))
    lambda_trans_p2 = float(config.get("lambda_transition_phase2", 5.0))
    tau = float(config.get("transition_tau", 0.05))
    val_split = float(config.get("val_split", 0.15))
    seed = int(config.get("seed", 42))
    warmup_ratio = float(config.get("warmup_ratio", 0.1))

    epochs = int(config.get("epochs", 0))
    if epochs > 0:
        phase1_epochs = max(2, epochs // 3)
        phase2_epochs = epochs - phase1_epochs

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading training data from {data_path}...")
    all_records = _load_training_data(data_path)
    print(f"Loaded {len(all_records)} valid records")

    base_ids = sorted(set(r.get("base_sample_id", r.get("sample_id", "")) for r in all_records))
    rng = random.Random(seed)
    rng.shuffle(base_ids)
    val_count = max(1, int(len(base_ids) * val_split))
    val_ids = set(base_ids[:val_count])
    train_ids = set(base_ids[val_count:])

    train_records = [r for r in all_records if r.get("base_sample_id", r.get("sample_id", "")) in train_ids]
    val_records = [r for r in all_records if r.get("base_sample_id", r.get("sample_id", "")) in val_ids]
    print(f"Train: {len(train_records)} records ({len(train_ids)} base samples)")
    print(f"Val:   {len(val_records)} records ({len(val_ids)} base samples)")

    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone_config = AutoConfig.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name, config=backbone_config)
    model = _SurrogateHead(backbone, backbone_config.hidden_size).to(device)

    mixed_dataset = _PairDataset(train_records, tokenizer, max_len)
    pair_dataset = _PairOnlyDataset(train_records, tokenizer, max_len)
    val_dataset = _PairDataset(val_records, tokenizer, max_len)

    mixed_loader = torch.utils.data.DataLoader(
        mixed_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
    )
    pair_loader = torch.utils.data.DataLoader(
        pair_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )

    print(f"Mixed dataset: {len(mixed_dataset)} items ({mixed_dataset._pair_count} pairs)")
    print(f"Pair-only dataset: {len(pair_dataset)} items")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_epochs = phase1_epochs + phase2_epochs
    total_steps = phase1_epochs * len(mixed_loader) + phase2_epochs * len(pair_loader)
    warmup_steps = int(total_steps * warmup_ratio)

    def _lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_schedule)

    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    best_val_score = float("inf")
    training_log: List[Dict[str, float]] = []
    global_step = 0

    print(f"\n=== Phase 1: Probability focus ({phase1_epochs} epochs, lambda_trans={lambda_trans_p1}) ===")
    print(f"=== Phase 2: Transition focus ({phase2_epochs} epochs, lambda_trans={lambda_trans_p2}, pair-only) ===")
    print(f"Total: {total_epochs} epochs, {total_steps} steps\n")

    for epoch in range(total_epochs):
        is_phase2 = epoch >= phase1_epochs
        phase_name = "P2" if is_phase2 else "P1"
        loader = pair_loader if is_phase2 else mixed_loader
        lambda_trans = lambda_trans_p2 if is_phase2 else lambda_trans_p1

        model.train()
        epoch_loss = 0.0
        epoch_prob_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_trans_loss = 0.0
        steps = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            prob_target = batch["prob_target"].to(device, non_blocking=True)
            class_target = batch["class_target"].to(device, non_blocking=True)
            is_pair = batch["is_pair"].to(device, non_blocking=True)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_prob = torch.sigmoid(out.logits.squeeze(-1))

            loss_prob = mse_loss_fn(pred_prob, prob_target)
            loss_cls = ce_loss_fn(out.class_logits, class_target) if not is_phase2 else torch.tensor(0.0, device=device)

            loss_trans = torch.tensor(0.0, device=device)
            pair_mask = is_pair.bool()
            cpu_pair_mask = batch["is_pair"].bool()
            if pair_mask.any():
                pair_ids = batch["pair_input_ids"][cpu_pair_mask].to(device, non_blocking=True)
                pair_am = batch["pair_attention_mask"][cpu_pair_mask].to(device, non_blocking=True)
                pair_target = batch["pair_prob_target"][cpu_pair_mask].to(device, non_blocking=True)

                pair_out = model(input_ids=pair_ids, attention_mask=pair_am)
                pair_pred = torch.sigmoid(pair_out.logits.squeeze(-1))

                loss_trans = _compute_transition_loss(
                    pred_prob[pair_mask], pair_pred,
                    prob_target[pair_mask], pair_target,
                    tau=tau,
                )

            w_prob = 0.3 if is_phase2 else 1.0
            loss = w_prob * loss_prob + lambda_cls * loss_cls + lambda_trans * loss_trans

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += loss.item()
            epoch_prob_loss += loss_prob.item()
            epoch_cls_loss += loss_cls.item()
            epoch_trans_loss += loss_trans.item()
            steps += 1

        avg_loss = epoch_loss / max(1, steps)
        avg_prob = epoch_prob_loss / max(1, steps)
        avg_cls = epoch_cls_loss / max(1, steps)
        avg_trans = epoch_trans_loss / max(1, steps)

        val_metrics = _evaluate(model, val_loader, device, tau)

        log_entry = {
            "epoch": epoch + 1,
            "phase": phase_name,
            "train_loss": avg_loss,
            "train_prob_loss": avg_prob,
            "train_cls_loss": avg_cls,
            "train_trans_loss": avg_trans,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        training_log.append(log_entry)

        val_score = val_metrics["mae"] * (1.0 - val_metrics["transition_agreement"])
        print(
            f"  [{phase_name}] Epoch {epoch+1}/{total_epochs}: "
            f"loss={avg_loss:.4f} (prob={avg_prob:.4f} cls={avg_cls:.4f} trans={avg_trans:.4f}) | "
            f"val_mae={val_metrics['mae']:.4f} val_trans={val_metrics['transition_agreement']:.4f}"
        )

        if val_score < best_val_score:
            best_val_score = val_score
            _save_checkpoint(model, tokenizer, backbone_config, out_path, val_metrics, config)
            print(
                f"    -> Best checkpoint (MAE={val_metrics['mae']:.4f}, "
                f"trans={val_metrics['transition_agreement']:.4f}, "
                f"score={val_score:.6f})"
            )

    final_meta = {
        "model_name": model_name,
        "best_val_score": best_val_score,
        "training_config": config,
        "training_log": training_log,
        "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_path / "training_summary.json").write_text(
        json.dumps(final_meta, indent=2), encoding="utf-8"
    )

    try:
        from modal import Volume
        Volume.from_name("gptzero-audit-artifacts").commit()
    except Exception:
        pass

    print(f"\nTraining complete. Best val score: {best_val_score:.6f}")
    print(f"Model saved to {out_path}")

    return final_meta


def _evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    margin: float,
) -> Dict[str, float]:
    model.eval()
    all_pred: List[float] = []
    all_true: List[float] = []
    all_pairs_base_pred: List[float] = []
    all_pairs_var_pred: List[float] = []
    all_pairs_base_true: List[float] = []
    all_pairs_var_true: List[float] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prob_target = batch["prob_target"]

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_prob = torch.sigmoid(out.logits.squeeze(-1)).cpu().numpy()

            all_pred.extend(pred_prob.tolist())
            all_true.extend(prob_target.numpy().tolist())

            pair_mask = batch["is_pair"].bool()
            if pair_mask.any():
                pair_ids = batch["pair_input_ids"][pair_mask].to(device)
                pair_am = batch["pair_attention_mask"][pair_mask].to(device)
                pair_target = batch["pair_prob_target"][pair_mask]

                pair_out = model(input_ids=pair_ids, attention_mask=pair_am)
                pair_pred = torch.sigmoid(pair_out.logits.squeeze(-1)).cpu().numpy()

                base_pred_p = pred_prob[pair_mask.numpy()]
                base_true_p = prob_target[pair_mask].numpy()

                all_pairs_base_pred.extend(base_pred_p.tolist())
                all_pairs_var_pred.extend(pair_pred.tolist())
                all_pairs_base_true.extend(base_true_p.tolist())
                all_pairs_var_true.extend(pair_target.numpy().tolist())

    pred_arr = np.array(all_pred)
    true_arr = np.array(all_true)
    mae = float(np.mean(np.abs(pred_arr - true_arr)))
    mse = float(np.mean((pred_arr - true_arr) ** 2))

    transition_agreement = 0.0
    if all_pairs_base_pred:
        bp = np.array(all_pairs_base_pred)
        vp = np.array(all_pairs_var_pred)
        bt = np.array(all_pairs_base_true)
        vt = np.array(all_pairs_var_true)
        delta_pred = vp - bp
        delta_true = vt - bt
        sign_match = np.sign(delta_pred) == np.sign(delta_true)
        near_zero = np.abs(delta_true) < 0.005
        valid = ~near_zero
        if valid.sum() > 0:
            transition_agreement = float(sign_match[valid].mean())

    pred_class = np.where(pred_arr >= 0.65, 2, np.where(pred_arr <= 0.35, 0, 1))
    true_class = np.where(true_arr >= 0.65, 2, np.where(true_arr <= 0.35, 0, 1))
    class_agreement = float((pred_class == true_class).mean())

    return {
        "mae": mae,
        "mse": mse,
        "transition_agreement": transition_agreement,
        "class_agreement": class_agreement,
        "pair_count": len(all_pairs_base_pred),
        "sample_count": len(all_pred),
    }


def _save_checkpoint(
    model: nn.Module,
    tokenizer: Any,
    config: Any,
    out_path: Path,
    metrics: Dict[str, float],
    train_config: Dict[str, Any],
) -> None:
    tokenizer.save_pretrained(str(out_path))
    config.save_pretrained(str(out_path))
    model.backbone.save_pretrained(str(out_path))
    head_state = {
        "prob_head": model.prob_head.state_dict(),
        "class_head": model.class_head.state_dict(),
    }
    torch.save(head_state, str(out_path / "surrogate_head.pt"))

    meta = {
        "model_type": "neural_surrogate",
        "backbone": str(train_config.get("model_name", DEFAULT_MODEL_NAME)),
        "val_metrics": metrics,
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_path / "surrogate_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
