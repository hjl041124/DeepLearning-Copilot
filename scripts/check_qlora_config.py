import json
from pathlib import Path


path = Path(
    "configs/qlora_config_v1.json"
)


data = json.loads(
    path.read_text(
        encoding="utf-8"
    )
)


errors=[]


if not data["quantization"]["load_in_4bit"]:
    errors.append(
        "4bit quantization disabled"
    )


if data["lora"]["r"] <= 0:
    errors.append(
        "invalid lora rank"
    )


required_modules = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
}


if set(
    data["lora"]["target_modules"]
) != required_modules:
    errors.append(
        "target modules mismatch"
    )


if errors:

    print(
        "QLORA CONFIG VALIDATION FAILED"
    )

    for e in errors:
        print("-",e)

    raise SystemExit(1)


print(
    "QLORA CONFIG VALIDATION PASSED"
)

print(
    "LoRA rank:",
    data["lora"]["r"]
)

print(
    "Target modules:",
    data["lora"]["target_modules"]
)

print(
    "4bit:",
    data["quantization"]["load_in_4bit"]
)
