import time
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "models/Qwen3-4B-Instruct-2507"


def main():
    print("========================================")
    print("DeepLearning-Copilot")
    print("DLC-D1-E02 Base Model Smoke Test")
    print("========================================")

    print("\n===== ENVIRONMENT =====")
    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model path:", MODEL_PATH)

    print("\n===== LOAD TOKENIZER =====")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
    print("Tokenizer loaded successfully.")

    print("\n===== LOAD MODEL =====")
    load_start = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )

    model.eval()

    load_time = time.time() - load_start

    print(f"Model loaded successfully in {load_time:.2f} seconds.")
    print("Model device:", next(model.parameters()).device)
    print(
        "GPU memory allocated after loading (GB):",
        round(torch.cuda.memory_allocated() / 1024**3, 3),
    )

    messages = [
        {
            "role": "user",
            "content": (
                "In one or two sentences, explain what overfitting means "
                "in a deep learning experiment."
            ),
        }
    ]

    print("\n===== BUILD CHAT INPUT =====")

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    print("Input tokens:", inputs["input_ids"].shape[-1])

    print("\n===== GENERATION =====")

    torch.cuda.reset_peak_memory_stats()

    generation_start = time.time()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
        )

    generation_time = time.time() - generation_start

    generated_ids = outputs[
        0,
        inputs["input_ids"].shape[-1]:
    ]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print("\n===== MODEL RESPONSE =====")
    print(response)

    print("\n===== RUNTIME =====")
    print(f"Generation time: {generation_time:.2f} seconds.")
    print(
        "Peak GPU memory during generation (GB):",
        round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    )

    print("\n===== RESULT =====")

    if response.strip():
        print("SMOKE TEST PASSED")
    else:
        raise RuntimeError("Model generated an empty response.")


if __name__ == "__main__":
    main()
