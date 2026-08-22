"""Lazy, process-reusable wrapper for the final QLoRA diagnosis model."""

import json
from pathlib import Path
from threading import Lock
from typing import Any


DEFAULT_BASE_MODEL = "models/Qwen3-4B-Instruct-2507"
DEFAULT_ADAPTER_PATH = "models/qwen3-4b-dlcopilot-final-qlora"


class QLoRADiagnosisModel:
    """Load the final model on first use and reuse it in this process."""

    def __init__(
        self,
        base_model_path: str = DEFAULT_BASE_MODEL,
        adapter_path: str = DEFAULT_ADAPTER_PATH,
    ) -> None:
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._load_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self.is_loaded:
            return

        with self._load_lock:
            if self.is_loaded:
                return

            if not Path(self.base_model_path).exists():
                raise FileNotFoundError(
                    f"base model not found: {self.base_model_path}"
                )

            if not Path(self.adapter_path).exists():
                raise FileNotFoundError(
                    f"QLoRA adapter not found: {self.adapter_path}"
                )

            import torch
            from peft import PeftModel
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path
            )

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                quantization_config=quantization_config,
                device_map="auto",
            )

            model = PeftModel.from_pretrained(
                base_model,
                self.adapter_path,
            )
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._torch = torch

    def generate(self, combined_context: dict[str, Any]) -> str:
        """Generate one deterministic structured diagnosis response."""

        if not isinstance(combined_context, dict):
            raise TypeError("combined_context must be a dictionary")

        self._load()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are DeepLearning-Copilot. Diagnose the experiment "
                    "only from the supplied structured evidence. Output JSON "
                    "only, using the canonical project vocabulary and exactly "
                    "these fields: task_type, primary_issue, severity, "
                    "evidence_codes, recommended_action_codes, explanation."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    combined_context,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {
            key: value.to(self._model.device)
            for key, value in inputs.items()
        }

        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )
