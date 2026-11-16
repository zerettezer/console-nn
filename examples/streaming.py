"""Feed a growing log in chunks and print the evolving verdict."""

from __future__ import annotations

from pathlib import Path

from console_nn import ConsoleAnalyzer


def main() -> None:
    path = Path(__file__).with_name("logs") / "02-pytest-failure.log"
    text = path.read_text(encoding="utf-8")

    stream = ConsoleAnalyzer().stream()
    for start in range(0, len(text), 256):
        result = stream.feed(text[start : start + 256])
        print(
            f"{start + 256:4d} chars  status={result.status:7s} "
            f"confidence={result.confidence:.3f}"
        )

    result = stream.flush()
    print(f"final: {result.status} completed={result.completed}")
    print(f"mean chunk latency: {stream.mean_chunk_latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
