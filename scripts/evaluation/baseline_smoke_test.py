import json
from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


MODEL_PATH = (
    "models/Qwen3-4B-Instruct-2507"
)


DATA_PATH = (
    "data/generated/full_test_v1.jsonl"
)


def load_first_sample():

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.loads(
            next(f)
        )


def main():

    print(
        "===== Baseline Smoke Test ====="
    )

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype="auto",
            device_map="auto",
        )
    )

    print(
        "Model loaded"
    )


    sample = load_first_sample()


    messages = sample[
        "messages"
    ]


    prompt = tokenizer.apply_chat_template(
        messages[:2],
        tokenize=False,
        add_generation_prompt=True,
    )


    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )


    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }


    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
    )


    generated = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]


    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )


    print()
    print(
        "===== MODEL OUTPUT ====="
    )

    print(text)


    print()
    print(
        "BASELINE SMOKE TEST PASSED"
    )


if __name__ == "__main__":
    main()
