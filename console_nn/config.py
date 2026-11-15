"""Fully serializable configuration for the console-log FCN.

Every tunable element of the pipeline lives here.  The config is stored as
plain JSON next to the trained artifacts, so a run can always be rebuilt
exactly.  Dotted overrides are supported on the CLI::

    python -m console_nn train --set train.epochs=40 --set model.filters=128
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------
@dataclass
class TokenizerConfig:
    # Vocabulary is hard-capped at ``max_vocab`` total ids, including
    # specials.  Set hash_buckets > 0 to spend part of that budget on
    # deterministic character-ngram buckets for OOV words.
    max_vocab: int = 5000
    min_freq: int = 2
    ngram_min: int = 1
    ngram_max: int = 3
    hash_buckets: int = 0          # extra OOV buckets (0 = map OOV to <unk>)
    hash_ngram: int = 3            # char ngram size used for hashing
    lowercase: bool = True
    normalize_numbers: bool = True  # mask mode maps every number to <num>
    # digits: 8080 -> <NUM> <8> <0> <8> <0>; bucket: magnitude classes;
    # keep: raw token; mask: one <num> token.
    numeric_mode: str = "digits"
    normalize_paths: bool = True    # /home/x/a.py -> <path>
    normalize_ips: bool = True
    strip_ansi: bool = True
    collapse_spaces: bool = True
    max_line_tokens: int = 128      # safety valve for one pathological line
    sep_token: str = "[SEP]"
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    bos_token: str = "[BOS]"
    eos_token: str = "[EOS]"
    mask_token: str = "[MASK]"
    # Add BOS/EOS around each line as a cheap structural signal.
    line_boundaries: bool = True
    # Trailing SEP after every line also marks boundaries.
    append_sep: bool = True


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
@dataclass
class EmbeddingConfig:
    # ALBERT-style factorised embedding: V x E followed by E x H.
    factorized: bool = True
    embedding_dim: int = 64        # E
    hidden_dim: int = 96           # H
    # "learned_chunk" = 16 learned positions shared by every chunk,
    # "sinusoidal"    = fixed sin/cos, works for any position,
    # "none"          = no positional signal (bag of ngrams inside a chunk).
    positional: str = "learned_chunk"
    max_chunk_pos: int = 16
    dropout: float = 0.05
    layer_norm: bool = True
    norm: str = "layer"            # layer | rms | none
    scale: float = 1.0             # multiply embeddings by this after projection


@dataclass
class ConvConfig:
    blocks: int = 4                # number of dilated residual blocks
    layers_per_block: int = 2
    filters: int = 96
    kernel_size: int = 3
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    conv_type: str = "separable"   # "separable" | "full"
    activation: str = "gelu"       # gelu | relu | swish | glu handled by gate
    gate: bool = True              # GLU on every conv
    norm: str = "layer"            # layer | batch | rms | none
    residual: bool = True
    dropout: float = 0.1
    l2: float = 0.0
    use_bias: bool = True
    padding: str = "same"


@dataclass
class PoolConfig:
    chunk_pool: str = "mean_max"   # mean | max | mean_max
    doc_pool: str = "mean_max"
    doc_use_chunk_probs: bool = True
    doc_hidden: int = 64
    dropout: float = 0.1


@dataclass
class HeadConfig:
    chunk_classes: int = 4
    doc_classes: int = 4
    class_names: List[str] = field(
        default_factory=lambda: ["success", "fail", "block", "unknown"]
    )
    chunk_hidden: int = 0          # 0 = direct linear from pooled features
    doc_hidden: int = 64
    head_activation: str = "gelu"  # relu is friendlier for int8
    completed_head: bool = True    # binary "operation reached a terminal state"


@dataclass
class ModelConfig:
    chunk_size: int = 16
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    conv: ConvConfig = field(default_factory=ConvConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)
    heads: HeadConfig = field(default_factory=HeadConfig)
    # 0.0 means no L2; optimizer weight decay is in TrainConfig.
    l2_embedding: float = 0.0


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@dataclass
class DataConfig:
    chunk_size: int = 16           # must match model.chunk_size
    train_stride: int = 16         # overlap augmentation when < chunk_size
    eval_stride: int = 16
    # 0 = keep all chunks, no document truncation (the architecture has no
    # sequence length limit, this only bounds RAM during training).
    max_chunks_per_doc: int = 512
    num_parallel_calls: int = 4
    prefetch: int = 4
    shuffle_buffer: int = 4096
    seed: int = 1337
    # chunk label priority when a chunk contains several annotated lines
    label_priority: List[str] = field(
        default_factory=lambda: ["fail", "block", "success", "unknown"]
    )
    # Lines without an explicit marker inherit the document status when the
    # operation succeeded, otherwise they stay "unknown".
    inherit_success_context: bool = True


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-4
    optimizer: str = "adamw"       # adamw | adam | sgd
    clipnorm: float = 1.0
    label_smoothing: float = 0.0
    loss_chunk: float = 1.0
    loss_doc: float = 1.0
    loss_completed: float = 0.5
    class_weights: List[float] = field(default_factory=list)  # per chunk class
    doc_class_weights: List[float] = field(default_factory=list)
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1.0e-5
    steps_per_epoch: int = 0        # 0 = full dataset
    validation_steps: int = 0
    mixed_precision: bool = False   # float16 compute on GPU
    jit_compile: bool = False
    verbose: int = 1
    log_csv: str = "history.csv"


@dataclass
class InferenceConfig:
    # 0 = process the whole document in one pass (no length limit).
    # A positive value bounds RAM for extremely long inputs.
    window_chunks: int = 0
    window_stride: int = 0
    batch_windows: int = 1
    tflite_threads: int = 4
    benchmark_chunks: int = 128    # only for latency measurement
    # How document-level predictions from overlapping windows are combined.
    doc_aggregate: str = "mean"    # mean | max_severity
    color: str = "auto"            # auto | always | never
    bar_width: int = 24
    max_line_chars: int = 160


@dataclass
class QuantConfig:
    mode: str = "int8"             # int8 (weights) | float16 | none
    denylisted_ops: List[str] = field(default_factory=list)
    verify: bool = True
    verify_samples: int = 0        # 0 = all documents in the split
    verify_tolerance: float = 0.05


@dataclass
class SynthConfig:
    n_train: int = 24000
    n_val: int = 3000
    n_test: int = 3000
    seed: int = 42
    platforms: List[str] = field(default_factory=lambda: ["linux", "windows"])
    domains: List[str] = field(
        default_factory=lambda: [
            "python", "pip", "pytest", "conda", "shell", "package", "systemd",
            "network", "filesystem", "git", "docker", "kernel", "journal",
            "build", "windows_cmd", "powershell", "windows_update", "security",
            "package_update", "compile", "media", "data_process", "archive",
            "backup", "database", "cloud", "commands", "errors", "real_errors",
            "multi", "multi_chain",
        ]
    )
    status_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "success": 0.36, "fail": 0.34, "block": 0.15, "unknown": 0.15
        }
    )
    truncation_rate: float = 0.10
    ansi_rate: float = 0.30
    timestamp_rate: float = 0.35
    noise_rate: float = 0.12
    hard_negative_rate: float = 0.20


@dataclass
class Config:
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    synth: SynthConfig = field(default_factory=SynthConfig)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _from_dict(cls, data: Dict[str, Any]):
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) and isinstance(value, dict):  # pragma: no cover
            kwargs[f.name] = _from_dict(f.type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def nested_update(obj: Any, data: Dict[str, Any]) -> Any:
    """Update a dataclass tree from a dict, ignoring unknown keys."""
    for f in fields(obj):
        if f.name not in data:
            continue
        value = data[f.name]
        current = getattr(obj, f.name)
        if is_dataclass(current) and isinstance(value, dict):
            nested_update(current, value)
        else:
            setattr(obj, f.name, value)
    return obj


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    cfg = Config()
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            nested_update(cfg, json.load(fh))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        parts = key.split(".")
        node: Any = cfg
        for part in parts[:-1]:
            node = getattr(node, part)
        old = getattr(node, parts[-1])
        setattr(node, parts[-1], _parse_value(raw, old))
    validate_config(cfg)
    return cfg


def _parse_value(raw: str, old: Any) -> Any:
    if isinstance(old, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(old, int):
        return int(raw)
    if isinstance(old, float):
        return float(raw)
    if isinstance(old, list):
        raw = raw.strip()
        if raw.startswith("["):
            return json.loads(raw)
        if not raw:
            return []
        parsed = []
        for part in raw.split(","):
            part = part.strip()
            if isinstance(old[0] if old else "", int):
                parsed.append(int(part))
            elif isinstance(old[0] if old else "", float):
                parsed.append(float(part))
            else:
                parsed.append(part)
        return parsed
    if isinstance(old, dict):
        return json.loads(raw)
    if old is None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def validate_config(cfg: Config) -> None:
    if cfg.model.chunk_size != cfg.data.chunk_size:
        raise ValueError(
            f"model.chunk_size ({cfg.model.chunk_size}) must equal "
            f"data.chunk_size ({cfg.data.chunk_size})"
        )
    if cfg.tokenizer.max_vocab < 256:
        raise ValueError("tokenizer.max_vocab must be >= 256")
    if cfg.tokenizer.hash_buckets < 0:
        raise ValueError("tokenizer.hash_buckets must be >= 0")
    if cfg.model.embedding.hidden_dim <= 0:
        raise ValueError("model.embedding.hidden_dim must be > 0")
    if cfg.model.conv.blocks < 1:
        raise ValueError("model.conv.blocks must be >= 1")
    if cfg.model.conv.conv_type not in ("separable", "full"):
        raise ValueError("model.conv.conv_type must be separable or full")
    if cfg.model.conv.norm not in ("layer", "batch", "rms", "none"):
        raise ValueError("model.conv.norm must be layer, batch, rms or none")
    if cfg.model.embedding.norm not in ("layer", "rms", "none"):
        raise ValueError("model.embedding.norm must be layer, rms or none")
    if cfg.model.pool.chunk_pool not in ("mean", "max", "mean_max"):
        raise ValueError("model.pool.chunk_pool must be mean, max or mean_max")
    if cfg.model.pool.doc_pool not in ("mean", "max", "mean_max"):
        raise ValueError("model.pool.doc_pool must be mean, max or mean_max")
    if len(cfg.model.heads.class_names) != cfg.model.heads.chunk_classes:
        raise ValueError("heads.class_names length must match chunk_classes")
    if cfg.quant.mode not in ("int8", "float16", "none"):
        raise ValueError("invalid quant.mode")
    if cfg.inference.window_stride <= 0:
        cfg.inference.window_stride = cfg.inference.window_chunks


def save_config(cfg: Config, path: str) -> None:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=2)
        fh.write("\n")


def config_dict(cfg: Config) -> Dict[str, Any]:
    return asdict(cfg)
