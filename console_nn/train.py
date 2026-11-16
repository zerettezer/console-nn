"""Training entry point."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from .config import Config, save_config
from .data import (
    encode_split,
    fit_tokenizer,
    load_documents,
    make_dataset,
)
from .evaluate import evaluate_dataset, save_json
from .model import build_model, compile_model, model_summary
from .synth import corpus_stats
from .tokenizer import NgramTokenizer


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu(memory_mb: int = 0) -> None:
    """Configure GPU allocation before the first TensorFlow GPU operation.

    ``0`` enables memory growth.  A positive value caps this process on every
    visible GPU, which makes small concurrent training jobs predictable.
    """
    limit = int(memory_mb or os.environ.get("CONSOLE_NN_GPU_MEMORY_MB", 0) or 0)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            if limit > 0:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=limit)],
                )
            else:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # TensorFlow was already initialised in this process.
            pass


def _history_to_dict(history) -> Dict[str, list]:
    return {k: [float(x) for x in v] for k, v in history.history.items()}


def train_model(
    cfg: Config,
    data_dir: str,
    out_dir: str,
    tokenizer: Optional[NgramTokenizer] = None,
    verbose: Optional[int] = None,
) -> Dict:
    run = Path(out_dir)
    model_dir = run / "model"
    logs = run / "logs"
    metrics_dir = run / "metrics"
    reports_dir = run / "reports"
    for folder in (run, model_dir, logs, metrics_dir, reports_dir):
        folder.mkdir(parents=True, exist_ok=True)
    verbose = cfg.train.verbose if verbose is None else verbose
    configure_gpu(cfg.train.gpu_memory_mb)
    set_seed(cfg.data.seed)
    save_config(cfg, str(model_dir / "config.json"))
    if cfg.train.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    t0 = time.time()
    train_docs = load_documents(data_dir, "train")
    val_docs = load_documents(data_dir, "val")
    test_docs = load_documents(data_dir, "test")

    if tokenizer is None:
        print(f"[train] fitting tokenizer on {len(train_docs)} documents ...")
        tokenizer = fit_tokenizer(train_docs, cfg)
    print(f"[train] vocab={tokenizer.vocab_size} ids, specials={tokenizer.specials}")
    tokenizer.save(str(model_dir / "tokenizer.json"))

    print("[train] encoding splits ...")
    train_enc = encode_split(train_docs, tokenizer, cfg, stride=cfg.data.train_stride)
    val_enc = encode_split(val_docs, tokenizer, cfg, stride=cfg.data.eval_stride)
    test_enc = encode_split(test_docs, tokenizer, cfg, stride=cfg.data.eval_stride)
    print(
        f"[train] chunks: train={sum(x[0].shape[0] for x in train_enc)} "
        f"val={sum(x[0].shape[0] for x in val_enc)} "
        f"test={sum(x[0].shape[0] for x in test_enc)}"
    )

    train_ds = make_dataset(
        train_enc, cfg, cfg.train.batch_size, shuffle=True, seed=cfg.data.seed
    )
    if cfg.train.steps_per_epoch:
        train_ds = train_ds.repeat()
    val_ds = make_dataset(val_enc, cfg, cfg.train.batch_size, shuffle=False)
    test_ds = make_dataset(test_enc, cfg, cfg.train.batch_size, shuffle=False)

    info = model_summary(cfg, tokenizer.vocab_size)
    print(
        f"[train] params={info['params']:,} "
        f"(fp32 ~{info['approx_fp32_mb']} MB, int8 ~{info['approx_int8_mb']} MB)"
    )

    model = build_model(cfg, tokenizer.vocab_size)
    compile_model(model, cfg)
    if verbose:
        model.summary()

    checkpoint = str(model_dir / "weights.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1 if verbose else 0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.train.early_stopping_patience,
            restore_best_weights=True,
            verbose=1 if verbose else 0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.train.reduce_lr_factor,
            patience=cfg.train.reduce_lr_patience,
            min_lr=cfg.train.min_learning_rate,
            verbose=1 if verbose else 0,
        ),
    ]

    fit_kwargs = dict(
        epochs=cfg.train.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=verbose,
    )
    if cfg.train.steps_per_epoch:
        fit_kwargs["steps_per_epoch"] = cfg.train.steps_per_epoch
    if cfg.train.validation_steps:
        fit_kwargs["validation_steps"] = cfg.train.validation_steps

    history = model.fit(train_ds, **fit_kwargs)
    model.save_weights(str(model_dir / "weights.h5"))
    if Path(checkpoint).exists():
        model.load_weights(checkpoint)

    save_json(_history_to_dict(history), str(metrics_dir / "history.json"))
    save_json(
        {
            "model": info,
            "tokenizer": tokenizer.stats,
            "data": {
                "train": corpus_stats(train_docs),
                "val": corpus_stats(val_docs),
                "test": corpus_stats(test_docs),
            },
        },
        str(reports_dir / "run_info.json"),
    )

    print("[train] evaluating validation ...")
    val_metrics = evaluate_dataset(
        model,
        val_ds,
        cfg.model.heads.class_names,
        cfg.model.heads.completed_head,
    )
    print("[train] evaluating test ...")
    test_metrics = evaluate_dataset(
        model,
        test_ds,
        cfg.model.heads.class_names,
        cfg.model.heads.completed_head,
    )
    save_json(val_metrics, str(metrics_dir / "val_metrics.json"))
    save_json(test_metrics, str(metrics_dir / "test_metrics.json"))

    summary = {
        "run_dir": str(run),
        "seconds": round(time.time() - t0, 1),
        "epochs_run": len(history.history.get("loss", [])),
        "model": info,
        "vocab_size": tokenizer.vocab_size,
        "val_doc_accuracy": val_metrics["doc"]["accuracy"],
        "val_doc_macro_f1": val_metrics["doc"]["macro_f1"],
        "test_doc_accuracy": test_metrics["doc"]["accuracy"],
        "test_doc_macro_f1": test_metrics["doc"]["macro_f1"],
        "test_chunk_accuracy": test_metrics["chunk"]["accuracy"],
        "test_chunk_macro_f1": test_metrics["chunk"]["macro_f1"],
        "test_completed": test_metrics.get("completed"),
    }
    save_json(summary, str(metrics_dir / "summary.json"))
    print(json.dumps(summary, indent=2))
    return summary
