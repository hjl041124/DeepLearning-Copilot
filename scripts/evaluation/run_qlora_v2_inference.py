import json
from pathlib import Path

import torch

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    PeftModel,
)


BASE_MODEL = (
    "models/Qwen3-4B-Instruct-2507"
)

ADAPTER_PATH = (
    "models/qwen3-4b-dlcopilot-qlora-v2"
)


DATASETS = {
    "standard_test":
        "data/generated/full_test_v1.jsonl",

    "hard_test":
        "data/generated/hard_test_v1.jsonl",
}


OUTPUT_DIR = Path(
    "reports/qlora"
)


def load_jsonl(path):

    data = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():

                data.append(
                    json.loads(line)
                )

    return data



def generate_one(
    model,
    tokenizer,
    sample,
):

    prompt = tokenizer.apply_chat_template(
        sample["messages"][:2],
        tokenize=False,
        add_generation_prompt=True,
    )


    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )


    inputs = {
        k:v.to(model.device)
        for k,v in inputs.items()
    }


    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )


    generated = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]


    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )



def main():

    print(
        "Loading tokenizer..."
    )


    tokenizer = (
        AutoTokenizer.from_pretrained(
            BASE_MODEL
        )
    )


    print(
        "Loading base model..."
    )


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


    base_model = (
        AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
        )
    )


    print(
        "Loading LoRA adapter..."
    )


    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )


    model.eval()


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    for name,path in DATASETS.items():

        samples = load_jsonl(
            path
        )

        output = (
            OUTPUT_DIR
            /
            f"{name}_predictions.jsonl"
        )


        with open(
            output,
            "w",
            encoding="utf-8",
        ) as f:


            for sample in tqdm(
                samples,
                desc=name,
            ):

                pred = generate_one(
                    model,
                    tokenizer,
                    sample,
                )


                f.write(
                    json.dumps(
                        {
                            "sample_id":
                                sample["sample_id"],

                            "prediction":
                                pred,

                            "ground_truth":
                                sample["ground_truth"],

                            "task_type":
                                sample["task_type"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


        print(
            "saved:",
            output
        )


    print(
        "QLORA INFERENCE PASSED"
    )


if __name__=="__main__":
    main()
