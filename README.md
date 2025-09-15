# 2025-iskyn

Minimal structure for running ONNX image inference with a simple CLI.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Predict with ONNX

```bash
python main.py --model best.onnx --image path/to/image.jpg --output out.png
```

`detect_objects` currently returns the input image unchanged. Implement your model-specific post-processing in `src/iskyn/inference/onnx_runtime.py`.


