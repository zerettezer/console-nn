"""N-gram tokenizer with a hard vocabulary budget (default 5000 ids).

The tokenizer is intentionally dependency free.  It works on normalized
console lines and emits unigrams, bigrams and trigrams.  Unknown n-grams are
either mapped to ``[UNK]`` or, when ``hash_buckets`` is enabled, to a
deterministic bucket inside the same vocabulary budget.
"""
from __future__ import annotations

import json
import zlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import TokenizerConfig
from .text import normalize_text, tokenize


class NgramTokenizer:
    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self.specials: List[str] = [
            self.config.pad_token,
            self.config.unk_token,
            self.config.bos_token,
            self.config.eos_token,
            self.config.sep_token,
            self.config.mask_token,
        ]
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        self.hash_base: int = 0
        self.hash_buckets: int = 0
        self.stats: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.config.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.config.unk_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.config.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.config.eos_token]

    @property
    def sep_id(self) -> int:
        return self.token_to_id[self.config.sep_token]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    # ------------------------------------------------------------------
    # normalization / ngrams
    # ------------------------------------------------------------------
    def normalize(self, text: str) -> str:
        c = self.config
        return normalize_text(
            text,
            lowercase=c.lowercase,
            normalize_numbers=c.normalize_numbers,
            normalize_paths=c.normalize_paths,
            normalize_ips=c.normalize_ips,
            collapse_spaces=c.collapse_spaces,
            numeric_mode=c.numeric_mode,
        )

    def ngrams(self, text: str) -> List[str]:
        """Return unigrams..n-grams for one line, in reading order."""
        c = self.config
        words = tokenize(self.normalize(text), self.config.numeric_mode)
        out: List[str] = []
        for n in range(c.ngram_min, c.ngram_max + 1):
            if n == 1:
                out.extend(words)
            else:
                for i in range(len(words) - n + 1):
                    out.append(" ".join(words[i : i + n]))
        return out

    def _hash_id(self, token: str) -> int:
        bucket = zlib.crc32(token.encode("utf-8")) % self.hash_buckets
        return self.hash_base + bucket

    def _id(self, token: str) -> int:
        tid = self.token_to_id.get(token)
        if tid is not None:
            return tid
        if self.hash_buckets > 0:
            return self._hash_id(token)
        return self.unk_id

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------
    def fit(self, documents: Iterable[Sequence[str]]) -> "NgramTokenizer":
        """Build the vocabulary from an iterable of documents (list of lines)."""
        c = self.config
        counts: Counter = Counter()
        total_lines = 0
        total_docs = 0
        for lines in documents:
            total_docs += 1
            for line in lines:
                total_lines += 1
                words = tokenize(self.normalize(line), c.numeric_mode)
                for n in range(c.ngram_min, c.ngram_max + 1):
                    if n == 1:
                        for w in words:
                            counts[w] += 1
                    else:
                        for i in range(len(words) - n + 1):
                            counts[" ".join(words[i : i + n])] += 1

        self.hash_buckets = max(0, min(c.hash_buckets, c.max_vocab // 4))
        budget = c.max_vocab - len(self.specials) - self.hash_buckets
        # Unigrams get a tiny boost so identifiers are not crowded out by
        # frequent boilerplate bigrams.
        candidates = []
        for token, count in counts.items():
            if count < c.min_freq:
                continue
            order = token.count(" ") + 1
            boost = {1: 1.15, 2: 1.0, 3: 0.9}.get(order, 0.8)
            candidates.append((count * boost, count, token))
        candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))

        vocab = list(self.specials)
        seen = set(vocab)
        for _, _, token in candidates:
            if len(vocab) >= len(self.specials) + budget:
                break
            if token in seen:
                continue
            vocab.append(token)
            seen.add(token)

        self.id_to_token = vocab
        self.token_to_id = {t: i for i, t in enumerate(vocab)}
        if self.hash_buckets:
            self.hash_base = len(vocab)
            self.id_to_token = vocab + [
                f"[HASH{ i }]" for i in range(self.hash_buckets)
            ]
            self.token_to_id = {t: i for i, t in enumerate(vocab)}

        self.stats = {
            "documents": total_docs,
            "lines": total_lines,
            "candidate_ngrams": len(counts),
            "kept_ngrams": len(vocab) - len(self.specials),
            "hash_buckets": self.hash_buckets,
            "vocab_size": len(self.id_to_token),
            "max_vocab": c.max_vocab,
            "ngram_range": [c.ngram_min, c.ngram_max],
        }
        return self

    # ------------------------------------------------------------------
    # encoding
    # ------------------------------------------------------------------
    def encode_tokens(self, tokens: Sequence[str]) -> List[int]:
        return [self._id(t) for t in tokens]

    def encode_line(self, text: str) -> Tuple[List[int], List[str]]:
        """Encode one line and return (ids, printable tokens)."""
        c = self.config
        toks: List[str] = []
        if c.line_boundaries:
            toks.append(c.bos_token)
        toks.extend(self.ngrams(text))
        if c.line_boundaries:
            toks.append(c.eos_token)
        if c.append_sep:
            toks.append(c.sep_token)
        if len(toks) > c.max_line_tokens:
            toks = toks[: c.max_line_tokens]
            if c.append_sep and toks:
                toks[-1] = c.sep_token
        return self.encode_tokens(toks), toks

    def encode_document(
        self, lines: Sequence[str]
    ) -> Tuple[List[int], List[int], List[str]]:
        """Encode every line, returning flat ids, source line ids and tokens."""
        all_ids: List[int] = []
        line_ids: List[int] = []
        all_tokens: List[str] = []
        for line_no, line in enumerate(lines):
            ids, toks = self.encode_line(line)
            all_ids.extend(ids)
            line_ids.extend([line_no] * len(ids))
            all_tokens.extend(toks)
        return all_ids, line_ids, all_tokens

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "id_to_token": self.id_to_token,
            "stats": self.stats,
            "hash_base": self.hash_base,
            "hash_buckets": self.hash_buckets,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
            fh.write("\n")

    @classmethod
    def load(cls, path: str) -> "NgramTokenizer":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        cfg = TokenizerConfig(**payload["config"])
        obj = cls(cfg)
        obj.id_to_token = list(payload["id_to_token"])
        obj.token_to_id = {t: i for i, t in enumerate(obj.id_to_token)}
        obj.stats = payload.get("stats", {})
        obj.hash_base = payload.get("hash_base", 0)
        obj.hash_buckets = payload.get("hash_buckets", 0)
        return obj

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> List[str]:
        out = []
        specials = set(self.specials)
        for i in ids:
            tok = self.id_to_token[int(i)] if 0 <= int(i) < len(self.id_to_token) else "<bad-id>"
            if skip_special and tok in specials:
                continue
            out.append(tok)
        return out
