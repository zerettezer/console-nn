# Usage examples

| file | purpose |
|---|---|
| `quickstart.py` | one log, one `LogAnalysis` result |
| `batch.py` | analyze a directory of logs and aggregate verdicts |
| `int8_vs_float.py` | compare float and int8 predictions/latency |
| `command_capture.py` | execute a command, capture output, classify it |
| `streaming.py` | feed a growing log in chunks with `StreamAnalyzer` |
| `custom_model.py` | load a specific packaged checkpoint |
| `cli_usage.sh` | CLI commands for reports, validation and deployment |

Visual logs/PNGs are in `logs/`; larger command/error corpora are in
`corpora/`. Curated Linux/Windows transcripts live in
`corpora/curated/real_transcripts.txt` and are built into the real corpus by
`tools/build_curated_corpus.py`.
