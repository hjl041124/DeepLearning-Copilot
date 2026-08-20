import json
from pathlib import Path

import torch

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


MODEL_PATH = (
    "models/Qwen3-4B-Instruct-2507"
)


DATASETS = {
    "standard_test":
        "data/generated/full_test_v1.jsonl",

    "hard_test":
        "data/generated/hard_test_v1.jsonl",
}


OUTPUT_DIR = Path(
    "reports/baseline"
)


GEN_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": False,
}


def load_jsonl(path):

    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def generate_one(
    model,
    tokenizer,
    sample,
):

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
        k: v.to(model.device)
        for k, v in inputs.items()
    }


    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            **GEN_CONFIG,
        )


    generated = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]


    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )


    return text


def main():

    print(
        "===== BASELINE INFERENCE ====="
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


    model.eval()


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    for name, path in DATASETS.items():

        print()

        print(
            "Running:",
            name,
        )


        samples = load_jsonl(
            path
        )


        output_path = (
            OUTPUT_DIR
            /
            f"{name}_predictions.jsonl"
        )


        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:


            for sample in tqdm(
                samples
            ):

                prediction = generate_one(
                    model,
                    tokenizer,
                    sample,
                )


                result = {
                    "sample_id":
                        sample[
                            "sample_id"
                        ],

                    "prediction":
                        prediction,

                    "ground_truth":
                        sample[
                            "ground_truth"
                        ],

                    "task_type":
                        sample[
                            "task_type"
                        ],

                    "scenario_family_id":
                        sample[
                            "scenario_family_id"
                        ]
                        if
                        "scenario_family_id"
                        in sample
                        else
                        sample[
                            "hard_test_family_id"
                        ],
                }


                f.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )


        print(
            "Saved:",
            output_path,
        )


    print(
        "BASELINE INFERENCE PASSED"
    )


if __name__ == "__main__":
    main()
