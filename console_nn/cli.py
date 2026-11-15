"""Command line interface."""
from __future__ import annotations

import argparse
import json
import os
import sys

# Keep TensorFlow's C++ INFO/WARNING chatter out of the user-facing log.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from pathlib import Path
from typing import List, Optional

from .config import config_dict, load_config, save_config


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="JSON config file")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, e.g. --set train.epochs=40",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="console-nn",
        description="Length-unlimited console-log analysis (FCN + N-grams).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # synth ---------------------------------------------------------
    p = sub.add_parser("synth", help="generate the synthetic corpus")
    p.add_argument("--out", default="data/synth")
    p.add_argument("--n", type=int, default=0, help="override train split size")
    _add_common(p)

    # train ---------------------------------------------------------
    p = sub.add_parser("train", help="train the FCN")
    p.add_argument("--data", default="data/synth")
    p.add_argument("--out", default="artifacts/default")
    _add_common(p)

    # eval ----------------------------------------------------------
    p = sub.add_parser("eval", help="evaluate a trained run")
    p.add_argument("--run", default="artifacts/default")
    p.add_argument("--data", default="data/synth")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    _add_common(p)

    # quantize ------------------------------------------------------
    p = sub.add_parser("quantize", help="export an int8/float16 TFLite model")
    p.add_argument("--run", default="artifacts/default")
    p.add_argument("--data", default="data/synth")
    p.add_argument("--mode", default=None,
                   choices=["int8", "float16", "none"])
    p.add_argument("--out", default=None)
    _add_common(p)

    # predict -------------------------------------------------------
    p = sub.add_parser("predict", help="analyze a log from a file/stdin/text/command")
    p.add_argument("--run", default=None, help="run directory (default: auto)")
    p.add_argument("-f", "--file", default=None, help="log file, '-' for stdin")
    p.add_argument("-t", "--text", default=None, help="inline text")
    p.add_argument("--exec", dest="exec_command", default=None,
                   help="run a command, capture stdout+stderr, analyze it")
    p.add_argument("--tflite", nargs="?", const=True, default=None,
                   help="use int8 TFLite (optionally a path)")
    p.add_argument("--format", default="ascii",
                   choices=["ascii", "brief", "json", "png"])
    p.add_argument("--details", action="store_true",
                   help="include the per-chunk probability table")
    p.add_argument("--out", default=None, help="write html/json to a file")
    p.add_argument("--color", default="auto", choices=["auto", "always", "never"])
    _add_common(p)

    # config --------------------------------------------------------
    p = sub.add_parser("config", help="print the default config")
    p.add_argument("--out", default=None)
    _add_common(p)

    # report --------------------------------------------------------
    p = sub.add_parser("report", help="consolidate run into log/metrics/benchmark/png")
    p.add_argument("--run", default="artifacts/cpu")
    p.add_argument("--clean", action="store_true",
                   help="remove intermediate logs/, metrics/, reports/ after bundling")

    # manifest / doctor ---------------------------------------------
    p = sub.add_parser("manifest", help="write checksums for a model folder")
    p.add_argument("--run", default="artifacts/cpu")

    p = sub.add_parser("doctor", help="deployment checks for a run")
    p.add_argument("--run", default="artifacts/cpu")
    p.add_argument("--data", default=None, help="optional real/eval data dir")
    p.add_argument("--strict", action="store_true")

    # validate ------------------------------------------------------
    p = sub.add_parser("validate", help="int8-vs-float validation")
    p.add_argument("--run", default="artifacts/cpu")
    p.add_argument("--data", default="data/synth")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--tflite", default=None)
    p.add_argument("--max-docs", type=int, default=0,
                   help="0 = every document in the split")
    p.add_argument("--min-doc-accuracy", type=float, default=0.85)
    p.add_argument("--max-accuracy-drop", type=float, default=0.10)
    p.add_argument("--strict", action="store_true",
                   help="exit 1 when the quality gate fails")

    # benchmark -----------------------------------------------------
    p = sub.add_parser("benchmark", help="measure CPU TFLite latency")
    p.add_argument("--run", default="artifacts/cpu")
    p.add_argument("--tflite", default=None)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--window-chunks", type=int, default=128,
                   help="chunk count used for latency measurement")
    p.add_argument("--save", action="store_true",
                   help="write metrics/speed.json and logs/speed.csv")

    # models --------------------------------------------------------
    p = sub.add_parser("models", help="list discoverable trained models")
    p.add_argument("--json", action="store_true")
    _add_common(p)

    # dataset -------------------------------------------------------
    p = sub.add_parser("dataset-stats", help="show statistics for a corpus")
    p.add_argument("--data", default="data/synth")
    _add_common(p)

    return parser


def cmd_synth(args) -> int:
    from .synth import write_corpus

    cfg = load_config(args.config, args.set)
    if args.n:
        cfg.synth.n_train = args.n
    meta = write_corpus(cfg.synth, args.out)
    save_config(cfg, str(Path(args.out) / "synth_config.json"))
    return 0


def cmd_train(args) -> int:
    from .train import train_model

    cfg = load_config(args.config, args.set)
    train_model(cfg, args.data, args.out)
    return 0


def cmd_eval(args) -> int:
    from .evaluate import evaluate_run

    metrics = evaluate_run(args.run, args.data, args.split)
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_quantize(args) -> int:
    from .quantize import quantize_run

    from .evaluate import resolve_model_dir

    cfg = load_config(str(resolve_model_dir(args.run) / "config.json"), args.set)
    quantize_run(args.run, args.data, mode=args.mode, out_path=args.out, cfg=cfg)
    return 0


def cmd_predict(args) -> int:
    from .api import ConsoleAnalyzer

    cfg = load_config(args.config, args.set)
    analyzer = ConsoleAnalyzer(model_dir=args.run, tflite=args.tflite)
    if args.exec_command:
        result = analyzer.analyze_command(args.exec_command)
    elif args.text is not None:
        result = analyzer.analyze(args.text)
    elif args.file:
        if args.file == "-":
            result = analyzer.analyze(sys.stdin.read())
        else:
            result = analyzer.analyze_file(args.file)
    else:
        if sys.stdin.isatty():
            print("[predict] reading log from stdin (Ctrl-D to finish) ...", file=sys.stderr)
        result = analyzer.analyze(sys.stdin.read())

    if args.format == "json":
        text = result.to_json(indent=2)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
    elif args.format == "png":
        target = args.out or "console_report.png"
        result.to_png(target)
        print(f"[predict] wrote {target}")
    elif args.format == "brief":
        color = None
        if args.color == "always":
            color = True
        elif args.color == "never":
            color = False
        print(result.to_brief(color=color, max_line_chars=cfg.inference.max_line_chars))
    else:
        color = None
        if args.color == "always":
            color = True
        elif args.color == "never":
            color = False
        print(
            result.to_ascii(
                color=color,
                width=cfg.inference.bar_width,
                max_line_chars=cfg.inference.max_line_chars,
                show_chunks=args.details,
            )
        )
        if result.meta:
            print(json.dumps(result.meta, ensure_ascii=False))
    return 0


def cmd_config(args) -> int:
    cfg = load_config(args.config, args.set)
    text = json.dumps(config_dict(cfg), indent=2)
    if args.out:
        save_config(cfg, args.out)
        print(f"[config] wrote {args.out}")
    else:
        print(text)
    return 0


def cmd_report(args) -> int:
    import shutil

    from .reporting import build_run_reports

    build_run_reports(args.run)
    if args.clean:
        from .evaluate import resolve_model_dir

        model_dir = resolve_model_dir(args.run)
        run = model_dir.parent if model_dir.name == "model" else model_dir
        for name in ("logs", "metrics", "reports"):
            shutil.rmtree(run / name, ignore_errors=True)
    print(f"[report] wrote {args.run}/run.log, metrics.json, benchmark.json")
    return 0


def cmd_manifest(args) -> int:
    from .doctor import write_manifest

    print(json.dumps(write_manifest(args.run), indent=2))
    return 0


def cmd_doctor(args) -> int:
    from .doctor import doctor

    report = doctor(args.run, data_dir=args.data)
    print(json.dumps(report, indent=2))
    if args.strict and not report["ok"]:
        return 1
    return 0


def cmd_validate(args) -> int:
    from .int8_validation import validate_run

    report = validate_run(
        args.run,
        args.data,
        split=args.split,
        max_docs=args.max_docs,
        tflite=args.tflite,
        save=True,
        min_doc_accuracy=args.min_doc_accuracy,
        max_accuracy_drop=args.max_accuracy_drop,
    )
    print(json.dumps(report, indent=2))
    if args.strict and not report["gates"]["ok"]:
        return 1
    return 0


def cmd_benchmark(args) -> int:
    from .benchmark import benchmark_run

    report = benchmark_run(
        args.run,
        tflite=args.tflite,
        iters=args.iters,
        threads=args.threads,
        window_chunks=args.window_chunks,
        save=args.save,
    )
    print(json.dumps(report, indent=2))
    return 0


def cmd_models(args) -> int:
    from .api import available_models

    models = available_models()
    if args.json:
        print(json.dumps(models, indent=2))
    elif not models:
        print("no models found")
    else:
        for m in models:
            print(f"{m['path']}")
            if "test_doc_accuracy" in m:
                print(
                    f"  doc_acc={m.get('test_doc_accuracy')} "
                    f"macro_f1={m.get('test_doc_macro_f1')} "
                    f"chunk_acc={m.get('test_chunk_accuracy')}"
                )
            if m.get("tflite"):
                print(f"  tflite: {', '.join(m['tflite'])}")
    return 0


def cmd_dataset_stats(args) -> int:
    from .synth import corpus_stats, read_jsonl

    path = Path(args.data)
    out = {"splits": {}}
    for split in ("train", "val", "test"):
        file = path / f"{split}.jsonl"
        if file.exists():
            out["splits"][split] = corpus_stats(read_jsonl(str(file)))
    out["total"] = sum(v["documents"] for v in out["splits"].values())
    print(json.dumps(out, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "synth": cmd_synth,
        "train": cmd_train,
        "eval": cmd_eval,
        "quantize": cmd_quantize,
        "predict": cmd_predict,
        "config": cmd_config,
        "report": cmd_report,
        "manifest": cmd_manifest,
        "doctor": cmd_doctor,
        "validate": cmd_validate,
        "benchmark": cmd_benchmark,
        "models": cmd_models,
        "dataset-stats": cmd_dataset_stats,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
