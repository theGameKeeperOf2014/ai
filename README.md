# No-API Chatbot AI

This project gives you a chatbot AI that **does not require an API key or sign-in**.
It downloads and runs a public transformer model directly from Hugging Face.

## Features

- No API keys
- No login/sign-in
- Runs locally on your machine
- Web chat UI (Gradio)
- Uses a transformer model via model ID ("transformer link" style)

## Quick Start

1. Install Python 3.10+
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start chatbot:

```bash
python app.py
```

4. Open in browser:

```text
http://localhost:7860
```

## Use another transformer model

You can pass any compatible Hugging Face causal language model ID:

```bash
python app.py --model microsoft/DialoGPT-medium
```

Examples:
- `microsoft/DialoGPT-small` (default, lighter)
- `microsoft/DialoGPT-medium` (better quality, more resources)

## Notes

- First run downloads model files from Hugging Face.
- No external chatbot API is used.
- If you have a GPU with CUDA, it will use it automatically.
