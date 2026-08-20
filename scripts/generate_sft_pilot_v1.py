import json
import random
import hashlib
from pathlib import Path


OUTPUT_PATH = Path(
    "data/generated/sft_pilot_v1.jsonl"
)


SOURCE_PATH = Path(
    "data/generated/full_dataset_v1.jsonl"
)


TARGET_COUNT = 2000


CATEGORY_RATIO = {
    "schema_alignment": 800,
    "evidence_grounding": 600,
    "recommendation_mapping": 400,
    "hard_reasoning": 200,
}


SYSTEM_PROMPT = """
You are DeepLearning-Copilot,
an AI assistant for deep learning experiment diagnosis.

You must output only valid JSON.

You must strictly follow:
- project task vocabulary
- evidence codes
- recommended action codes
- output schema

Do not use natural language labels.
"""


def load_jsonl(path):

    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():

                records.append(
                    json.loads(line)
                )

    return records



def make_id(prefix, idx):

    value = (
        f"{prefix}_{idx}"
    )

    return hashlib.md5(
        value.encode()
    ).hexdigest()[:12]



def build_record(
    source,
    category,
):

    gt = source[
        "ground_truth"
    ]

    user = (
        "Please diagnose this "
        "deep learning experiment.\n\n"
        +
        source["prompt"]
    )


    assistant = json.dumps(
        gt,
        ensure_ascii=False,
    )


    return {

        "id":
            make_id(
                category,
                random.randint(
                    0,
                    99999999
                )
            ),

        "category":
            category,

        "messages":[

            {
                "role":
                    "system",

                "content":
                    SYSTEM_PROMPT.strip()
            },

            {
                "role":
                    "user",

                "content":
                    user
            },

            {
                "role":
                    "assistant",

                "content":
                    assistant
            }
        ],

        "ground_truth":
            gt
    }



def main():

    random.seed(42)


    source = load_jsonl(
        SOURCE_PATH
    )


    diagnosis = [
        x
        for x in source
        if x["task_type"]
        ==
        "experiment_diagnosis"
    ]


    output = []


    categories = []

    for key, count in CATEGORY_RATIO.items():

        categories.extend(
            [key] * count
        )


    random.shuffle(
        categories
    )


    for idx, category in enumerate(
        categories
    ):

        sample = random.choice(
            diagnosis
        )

        output.append(
            build_record(
                sample,
                category,
            )
        )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        for item in output:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


    print(
        "SFT PILOT GENERATION PASSED"
    )

    print(
        "Samples:",
        len(output)
    )

    print(
        "Category Distribution:"
    )

    for k,v in CATEGORY_RATIO.items():

        print(
            k,
            v
        )


if __name__ == "__main__":
    main()
