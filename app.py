from __future__ import annotations

import argparse
from typing import List, Tuple

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(history: List[Tuple[str, str]], user_message: str) -> str:
    conversation = ["You are a helpful and friendly chatbot."]
    for user, assistant in history:
        conversation.append(f"User: {user}")
        conversation.append(f"Assistant: {assistant}")
    conversation.append(f"User: {user_message}")
    conversation.append("Assistant:")
    return "\n".join(conversation)


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model


def create_chat_fn(tokenizer, model, max_new_tokens: int, temperature: float, top_p: float):
    def chat(message: str, history: List[Tuple[str, str]]):
        history = history or []
        prompt = build_prompt(history, message)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = decoded[len(prompt) :].strip()
        if "User:" in answer:
            answer = answer.split("User:")[0].strip()
        return answer

    return chat


def main():
    parser = argparse.ArgumentParser(description="Local chatbot with no API key and no sign-in")
    parser.add_argument("--model", default="microsoft/DialoGPT-small", help="Hugging Face model ID")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model)
    chat_fn = create_chat_fn(
        tokenizer,
        model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    demo = gr.ChatInterface(
        fn=chat_fn,
        title="No-API Chatbot",
        description=(
            "Runs a transformer model directly from Hugging Face. "
            "No API key, no account sign-in required."
        ),
    )

    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
