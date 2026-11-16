"""Fully convolutional, length-unlimited console-log classifier.

Architecture
------------
tokens [B, C, 16]
  -> ALBERT-style factorised embedding (V x E -> E x H)
  -> shared intra-chunk positional embedding
  -> C chunks are folded into the batch: [B*C, 16, H]
  -> dilated, gated, depthwise-separable FCN (residual blocks)
  -> masked mean+max pooling inside every 16-token chunk
  -> chunk head      : 4 classes (success / fail / block / unknown)
  -> document pooling: masked mean+max over chunks (+ chunk probabilities)
  -> document head   : 4 classes
  -> completion head : binary "operation reached a terminal state"

There is no recurrent state and no fixed sequence length: the chunk count C
is dynamic, only the intra-chunk window is fixed at 16 tokens.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from .config import Config, ModelConfig


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="console_nn")
class FactorizedEmbedding(tf.keras.layers.Layer):
    """ALBERT-style factorised token embedding: V x E then E x H."""

    def __init__(self, vocab_size: int, cfg: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = int(vocab_size)
        self.cfg = cfg
        self.emb_dim = int(cfg.embedding.embedding_dim)
        self.hidden_dim = int(cfg.embedding.hidden_dim)

    def build(self, input_shape):
        c = self.cfg.embedding
        self.token_embedding = self.add_weight(
            name="token_embedding",
            shape=(self.vocab_size, self.emb_dim),
            initializer="truncated_normal",
            regularizer=tf.keras.regularizers.l2(self.cfg.l2_embedding)
            if self.cfg.l2_embedding
            else None,
        )
        if c.factorized:
            self.projection = self.add_weight(
                name="embedding_projection",
                shape=(self.emb_dim, self.hidden_dim),
                initializer="glorot_uniform",
                regularizer=tf.keras.regularizers.l2(self.cfg.l2_embedding)
                if self.cfg.l2_embedding
                else None,
            )
        else:
            self.projection = None
            if self.emb_dim != self.hidden_dim:
                raise ValueError(
                    "embedding.factorized=false requires embedding_dim == hidden_dim"
                )
        if c.positional == "learned_chunk":
            self.position = self.add_weight(
                name="position_embedding",
                shape=(c.max_chunk_pos, self.hidden_dim),
                initializer="truncated_normal",
            )
        elif c.positional == "sinusoidal":
            self.position = self._sinusoidal(c.max_chunk_pos, self.hidden_dim)
        elif c.positional == "none":
            self.position = None
        else:
            raise ValueError(f"unknown positional embedding {c.positional!r}")
        if c.layer_norm and c.norm == "rms":
            self.norm = RMSNorm()
        elif c.layer_norm and c.norm == "layer":
            self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        else:
            self.norm = None
        self.dropout = tf.keras.layers.Dropout(c.dropout)
        super().build(input_shape)

    @staticmethod
    def _sinusoidal(length: int, dim: int) -> tf.Tensor:
        pos = np.arange(length, dtype=np.float32)[:, None]
        i = np.arange(dim, dtype=np.float32)[None, :]
        angle = pos / np.power(10000.0, (2 * (i // 2)) / max(1, dim))
        values = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))
        return tf.constant(values.astype(np.float32))

    def call(self, inputs, training=None):
        ids = inputs                                   # [B, C, T]
        mask = tf.cast(tf.not_equal(ids, 0), tf.float32)[..., None]
        x = tf.gather(self.token_embedding, ids)       # [B, C, T, E]
        if self.projection is not None:
            x = tf.matmul(x, self.projection)          # [B, C, T, H]
        if self.cfg.embedding.scale != 1.0:
            x = x * tf.cast(self.cfg.embedding.scale, x.dtype)
        if self.position is not None:
            t = tf.shape(x)[2]
            pos = self.position[:t] if self.cfg.embedding.positional == "learned_chunk" \
                else self.position[:t]
            x = x + tf.cast(pos, x.dtype)[None, None, :, :]
        if self.norm is not None:
            x = self.norm(x)
        x = x * mask                                   # pads stay exactly zero
        x = self.dropout(x, training=training)
        return x


@tf.keras.utils.register_keras_serializable(package="console_nn")
class RMSNorm(tf.keras.layers.Layer):
    """Root-mean-square normalisation (no mean subtraction, no beta).

    Cheaper than LayerNorm and generally easier to quantize because there is
    no cross-channel mean/variance shift to preserve.
    """

    def __init__(self, epsilon: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma", shape=(int(input_shape[-1]),), initializer="ones"
        )
        super().build(input_shape)

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        scale = tf.math.rsqrt(
            tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.epsilon
        )
        return tf.cast(x * scale, inputs.dtype) * tf.cast(self.gamma, inputs.dtype)


@tf.keras.utils.register_keras_serializable(package="console_nn")
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, cfg: ModelConfig, dilation: int, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        c = cfg.conv
        self.filters = int(c.filters)
        self.gate = bool(c.gate)
        out_filters = self.filters * 2 if self.gate else self.filters
        reg = tf.keras.regularizers.l2(c.l2) if c.l2 else None
        if c.conv_type == "separable":
            conv_cls = tf.keras.layers.SeparableConv1D
            self.conv = conv_cls(
                out_filters,
                kernel_size=c.kernel_size,
                padding=c.padding,
                dilation_rate=dilation,
                depth_multiplier=1,
                use_bias=c.use_bias,
                depthwise_regularizer=reg,
                pointwise_regularizer=reg,
            )
        else:
            self.conv = tf.keras.layers.Conv1D(
                out_filters,
                kernel_size=c.kernel_size,
                padding=c.padding,
                dilation_rate=dilation,
                use_bias=c.use_bias,
                kernel_regularizer=reg,
            )
        if c.norm == "layer":
            self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        elif c.norm == "batch":
            self.norm = tf.keras.layers.BatchNormalization()
        elif c.norm == "rms":
            self.norm = RMSNorm()
        else:
            self.norm = None
        self.dropout = tf.keras.layers.Dropout(c.dropout)
        self.residual_proj = None
        if c.residual:
            self.residual_proj = tf.keras.layers.Conv1D(
                self.filters, 1, use_bias=False, name="residual_projection"
            )
        self.activation = c.activation

    def _act(self, x):
        if self.activation == "relu":
            return tf.nn.relu(x)
        if self.activation == "swish" or self.activation == "silu":
            return tf.nn.silu(x)
        return tf.nn.gelu(x)

    def call(self, inputs, training=None):
        h = inputs
        if self.norm is not None:
            h = self.norm(h)
        h = self.conv(h)
        if self.gate:
            a, b = tf.split(h, 2, axis=-1)
            h = a * tf.sigmoid(b)
        else:
            h = self._act(h)
        h = self.dropout(h, training=training)
        if self.cfg.conv.residual:
            residual = inputs if inputs.shape[-1] == self.filters \
                else self.residual_proj(inputs)
            h = h + residual
        return h


def _masked_pool(x, mask, mode: str, name: str):
    """x [N, T, F], mask [N, T] -> pooled [N, k*F]."""
    mask_f = tf.cast(mask, x.dtype)
    count = tf.reduce_sum(mask_f, axis=1, keepdims=True)
    denom = tf.maximum(count, 1.0)
    parts = []
    if mode in ("mean", "mean_max"):
        parts.append(tf.reduce_sum(x * mask_f[..., None], axis=1) / denom)
    if mode in ("max", "mean_max"):
        neg = tf.fill(tf.shape(x), tf.cast(-1e9, x.dtype))
        maxv = tf.reduce_max(tf.where(mask_f[..., None] > 0, x, neg), axis=1)
        # Fully padded rows would otherwise leak -1e9 into the next layer
        # (and blow up the TFLite int8 calibration range).
        maxv = tf.where(count > 0, maxv, tf.zeros_like(maxv))
        parts.append(maxv)
    if not parts:
        raise ValueError(mode)
    return tf.concat(parts, axis=-1, name=name)


@tf.keras.utils.register_keras_serializable(package="console_nn")
class ConsoleFCN(tf.keras.layers.Layer):
    def __init__(self, cfg: Config, vocab_size: int, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.vocab_size = int(vocab_size)
        self.chunk_size = int(cfg.model.chunk_size)
        self.embedding = FactorizedEmbedding(self.vocab_size, cfg.model)
        self.blocks = [
            ConvBlock(cfg.model, dilation=d)
            for d in _dilations(cfg.model.conv)
        ]
        self.chunk_hidden = None
        if cfg.model.heads.chunk_hidden > 0:
            self.chunk_hidden = tf.keras.layers.Dense(
                cfg.model.heads.chunk_hidden,
                activation=cfg.model.heads.head_activation,
            )
        self.chunk_dropout = tf.keras.layers.Dropout(cfg.model.pool.dropout)
        self.chunk_head = tf.keras.layers.Dense(
            cfg.model.heads.chunk_classes, name="chunk_logits"
        )
        doc_in = _pool_width(cfg.model.pool.doc_pool, cfg.model.embedding.hidden_dim)
        if cfg.model.pool.doc_use_chunk_probs:
            doc_in += _pool_width(
                cfg.model.pool.doc_pool, cfg.model.heads.chunk_classes
            )
        self.doc_dense = tf.keras.layers.Dense(
            cfg.model.heads.doc_hidden,
            activation=cfg.model.heads.head_activation,
        )
        self.doc_dropout = tf.keras.layers.Dropout(cfg.model.pool.dropout)
        self.doc_head = tf.keras.layers.Dense(
            cfg.model.heads.doc_classes, name="doc_logits"
        )
        self.completed_head = None
        if cfg.model.heads.completed_head:
            self.completed_head = tf.keras.layers.Dense(1, name="completed_logits")
        self._doc_in = doc_in

    def encode_chunks(self, ids, training=None):
        """Run the shared backbone and the chunk head.

        Returns ``(chunk_feat, chunk_logits, chunk_mask)`` with shapes
        ``[B, C, F]``, ``[B, C, K]`` and ``[B, C]``.  The document head is
        kept separate so a caller can pool chunks incrementally.
        """
        b = tf.shape(ids)[0]
        c = tf.shape(ids)[1]
        t = tf.shape(ids)[2]
        token_mask = tf.cast(tf.not_equal(ids, 0), tf.float32)

        x = self.embedding(ids, training=training)       # [B, C, T, H]
        h = tf.reshape(x, (-1, t, self.cfg.model.embedding.hidden_dim))
        flat_mask = tf.reshape(token_mask, (-1, t))
        for block in self.blocks:
            h = block(h, training=training)
        chunk_feat = _masked_pool(
            h, flat_mask, self.cfg.model.pool.chunk_pool, "chunk_pool"
        )
        if self.chunk_hidden is not None:
            chunk_feat = self.chunk_hidden(chunk_feat)
        chunk_feat = self.chunk_dropout(chunk_feat, training=training)
        chunk_logits = self.chunk_head(chunk_feat)       # [B*C, 4]
        classes = self.cfg.model.heads.chunk_classes
        chunk_logits = tf.reshape(chunk_logits, (-1, c, classes))

        chunk_mask = tf.reduce_max(token_mask, axis=2)   # [B, C]
        chunk_feat = tf.reshape(
            chunk_feat, (-1, c, tf.shape(chunk_feat)[-1])
        )
        return chunk_feat, chunk_logits, chunk_mask

    def document_logits(self, chunk_feat, chunk_mask, chunk_logits,
                        training=None):
        """Pool chunk features and run the document/completion heads."""
        doc_feat = _masked_pool(
            chunk_feat, chunk_mask, self.cfg.model.pool.doc_pool, "doc_pool"
        )
        if self.cfg.model.pool.doc_use_chunk_probs:
            probs = tf.nn.softmax(chunk_logits, axis=-1)
            prob_feat = _masked_pool(
                probs, chunk_mask, self.cfg.model.pool.doc_pool, "doc_prob_pool"
            )
            doc_feat = tf.concat([doc_feat, prob_feat], axis=-1)

        doc_feat = self.doc_dense(doc_feat)
        doc_feat = self.doc_dropout(doc_feat, training=training)
        doc_logits = self.doc_head(doc_feat)
        outputs = {
            "chunk_logits": chunk_logits,
            "doc_logits": doc_logits,
        }
        if self.completed_head is not None:
            outputs["completed_logits"] = self.completed_head(doc_feat)
        return outputs

    def call(self, inputs, training=None):
        chunk_feat, chunk_logits, chunk_mask = self.encode_chunks(
            inputs, training=training
        )
        return self.document_logits(
            chunk_feat, chunk_mask, chunk_logits, training=training
        )

    def get_config(self):
        return {"vocab_size": self.vocab_size}


def _dilations(cfg) -> list:
    out = []
    for b in range(cfg.blocks):
        for layer in range(cfg.layers_per_block):
            d = cfg.dilations[(b * cfg.layers_per_block + layer) % len(cfg.dilations)]
            out.append(d)
    return out


def _pool_width(mode: str, width: int) -> int:
    return width * 2 if mode == "mean_max" else width


def build_model(
    cfg: Config,
    vocab_size: int,
    fixed_chunks: Optional[int] = None,
) -> tf.keras.Model:
    if fixed_chunks is None:
        inputs = tf.keras.Input(
            shape=(None, cfg.model.chunk_size), dtype=tf.int32, name="tokens"
        )
    else:
        inputs = tf.keras.Input(
            shape=(int(fixed_chunks), cfg.model.chunk_size),
            dtype=tf.int32,
            name="tokens",
        )
    outputs = ConsoleFCN(cfg, vocab_size, name="console_fcn")(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="console_fcn_model")


# --------------------------------------------------------------------------
# losses / metrics
# --------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="console_nn")
class MaskedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    """Sparse CE that ignores chunk labels < 0 and supports class weights."""

    def __init__(
        self,
        class_weights=None,
        label_smoothing: float = 0.0,
        name: str = "masked_sparse_ce",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.class_weights = (
            tf.constant(class_weights, dtype=tf.float32)
            if class_weights is not None and len(class_weights) > 0
            else None
        )
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.cast(y_true > 0, tf.float32)
        safe = tf.where(y_true > 0, y_true - 1, tf.zeros_like(y_true))
        shape = tf.shape(y_pred)[:2]
        logp = tf.nn.log_softmax(y_pred, axis=-1)
        logp_flat = tf.reshape(logp, (-1, tf.shape(logp)[-1]))
        safe_flat = tf.reshape(safe, (-1,))
        nll = -tf.gather(logp_flat, safe_flat, batch_dims=1)
        nll = tf.reshape(nll, shape)
        if self.label_smoothing > 0:
            smooth = -tf.reduce_mean(logp_flat, axis=-1)
            smooth = tf.reshape(smooth, shape)
            nll = (1.0 - self.label_smoothing) * nll + self.label_smoothing * smooth
        if self.class_weights is not None:
            nll = nll * tf.gather(self.class_weights, safe, axis=0)
        nll = nll * mask
        return tf.reduce_sum(nll) / tf.maximum(tf.reduce_sum(mask), 1.0)

    def get_config(self):
        cfg = super().get_config()
        cfg["label_smoothing"] = self.label_smoothing
        return cfg


@tf.keras.utils.register_keras_serializable(package="console_nn")
class MaskedChunkAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name: str = "chunk_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.cast(y_true > 0, tf.float32)
        pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        safe = tf.where(y_true > 0, y_true - 1, tf.zeros_like(y_true))
        correct = tf.cast(tf.equal(pred, safe), tf.float32) * mask
        self.correct.assign_add(tf.reduce_sum(correct))
        self.total.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.correct / tf.maximum(self.total, 1.0)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


def compile_model(model: tf.keras.Model, cfg: Config) -> tf.keras.Model:
    t = cfg.train
    if t.optimizer == "adamw":
        opt = tf.keras.optimizers.experimental.AdamW(
            learning_rate=t.learning_rate, weight_decay=t.weight_decay
        )
    elif t.optimizer == "sgd":
        opt = tf.keras.optimizers.SGD(
            learning_rate=t.learning_rate, momentum=0.9, clipnorm=t.clipnorm
        )
    else:
        opt = tf.keras.optimizers.Adam(
            learning_rate=t.learning_rate, clipnorm=t.clipnorm
        )
    if t.optimizer != "sgd" and t.clipnorm:
        opt.clipnorm = t.clipnorm
    losses = {
        "chunk_logits": MaskedSparseCategoricalCrossentropy(
            class_weights=t.class_weights,
            label_smoothing=t.label_smoothing,
        ),
        "doc_logits": tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        "completed_logits": tf.keras.losses.BinaryCrossentropy(from_logits=True),
    }
    if cfg.model.heads.completed_head is False:
        losses.pop("completed_logits")
    # Shadow the doc loss with class weights if requested.
    if t.doc_class_weights:
        weights = tf.constant(t.doc_class_weights, dtype=tf.float32)

        class WeightedDocCE(tf.keras.losses.Loss):
            def call(self, y_true, y_pred):
                ce = tf.keras.losses.sparse_categorical_crossentropy(
                    tf.cast(y_true, tf.int32), y_pred, from_logits=True
                )
                return tf.reduce_mean(ce * tf.gather(weights, tf.cast(y_true, tf.int32)))

        losses["doc_logits"] = WeightedDocCE(name="weighted_doc_ce")

    model.compile(
        optimizer=opt,
        loss=losses,
        loss_weights={
            "chunk_logits": t.loss_chunk,
            "doc_logits": t.loss_doc,
            "completed_logits": t.loss_completed,
        }
        if cfg.model.heads.completed_head
        else {"chunk_logits": t.loss_chunk, "doc_logits": t.loss_doc},
        metrics={
            "chunk_logits": [MaskedChunkAccuracy()],
            "doc_logits": [tf.keras.metrics.SparseCategoricalAccuracy(name="doc_acc")],
            **(
                {
                    "completed_logits": [
                        tf.keras.metrics.BinaryAccuracy(name="completed_acc")
                    ]
                }
                if cfg.model.heads.completed_head
                else {}
            ),
        },
    )
    return model


def model_breakdown(model: tf.keras.Model) -> Dict[str, object]:
    """Parameter count per architectural block (helps judge model size)."""
    root = model.get_layer("console_fcn")
    embedding = int(root.embedding.count_params())
    conv = int(sum(block.count_params() for block in root.blocks))
    heads = int(
        sum(
            layer.count_params()
            for layer in (
                root.chunk_hidden,
                root.chunk_head,
                root.doc_dense,
                root.doc_head,
                root.completed_head,
            )
            if layer is not None
        )
    )
    total = int(model.count_params())
    out = {
        "total": total,
        "embedding": embedding,
        "conv": conv,
        "heads": heads,
        "other": total - embedding - conv - heads,
        "fp32_mb": round(total * 4 / 1024 / 1024, 3),
        "int8_mb": round(total / 1024 / 1024, 3),
    }
    out["embedding_share"] = round(embedding / max(1, total), 3)
    return out


def model_summary(cfg: Config, vocab_size: int) -> Dict[str, object]:
    model = build_model(cfg, vocab_size)
    breakdown = model_breakdown(model)
    return {
        "params": breakdown["total"],
        "approx_fp32_mb": round(breakdown["total"] * 4 / 1024 / 1024, 2),
        "approx_int8_mb": round(breakdown["total"] / 1024 / 1024, 2),
        "chunk_size": cfg.model.chunk_size,
        "vocab_size": vocab_size,
        "breakdown": breakdown,
    }
