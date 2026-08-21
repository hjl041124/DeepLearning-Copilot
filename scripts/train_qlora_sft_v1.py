import json
import argparse
from pathlib import Path

import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from trl import SFTTrainer


DEFAULT_CONFIG_PATH = Path(
    "configs/qlora_config_v1.json"
)


DATA_PATH = None


def load_config(config_path):

    return json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )


def load_dataset(data_path):

    records = []

    with open(
        data_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():

                item = json.loads(
                    line
                )

                records.append(
                    {
                        "messages":
                            item["messages"]
                    }
                )

    return Dataset.from_list(
        records
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
    )

    args = parser.parse_args()

    config = load_config(
        Path(args.config)
    )


    model_name = (
        "models/"
        + config["base_model"]
    )


    print(
        "Loading tokenizer..."
    )


    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_name
        )
    )


    print(
        "Loading 4-bit model..."
    )


    bnb_config = BitsAndBytesConfig(

        load_in_4bit=True,

        bnb_4bit_quant_type="nf4",

        bnb_4bit_compute_dtype=torch.bfloat16,

        bnb_4bit_use_double_quant=True,
    )


    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,

            quantization_config=bnb_config,

            device_map="auto",

            torch_dtype=torch.bfloat16,
        )
    )


    model.config.use_cache = False


    model = (
        prepare_model_for_kbit_training(
            model
        )
    )


    lora_config = LoraConfig(

        r=config["lora"]["r"],

        lora_alpha=
            config["lora"]["lora_alpha"],

        lora_dropout=
            config["lora"]["lora_dropout"],

        target_modules=
            config["lora"]["target_modules"],

        bias="none",

        task_type="CAUSAL_LM",
    )


    model = get_peft_model(
        model,
        lora_config,
    )


    model.print_trainable_parameters()


    dataset = load_dataset(Path(config["training"]["dataset"]))


    print(
        "Dataset size:",
        len(dataset)
    )


    training_args = TrainingArguments(

        output_dir=
            config["training"]["output_dir"],

        num_train_epochs=
            config["training"]["num_train_epochs"],

        per_device_train_batch_size=
            config["training"]["per_device_train_batch_size"],

        gradient_accumulation_steps=
            config["training"]["gradient_accumulation_steps"],

        learning_rate=
            config["training"]["learning_rate"],

        logging_steps=
            config["training"]["logging_steps"],

        save_steps=
            config["training"]["save_steps"],

        bf16=True,

        report_to="none",
    )


    trainer = SFTTrainer(

        model=model,

        args=training_args,

        train_dataset=dataset,

        processing_class=tokenizer,
    )


    print(
        "Starting QLoRA training..."
    )


    trainer.train()


    print(
        "Saving adapter..."
    )


    trainer.model.save_pretrained(
        config["training"]["output_dir"]
    )


    tokenizer.save_pretrained(
        config["training"]["output_dir"]
    )


    print(
        "QLORA TRAINING PASSED"
    )


if __name__ == "__main__":
    main()
