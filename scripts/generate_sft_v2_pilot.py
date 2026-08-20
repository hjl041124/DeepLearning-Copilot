import json
import random
import hashlib
from pathlib import Path


OUTPUT = Path(
    "data/generated/sft_v2_pilot.jsonl"
)


SOURCE = Path(
    "data/generated/full_dataset_v1.jsonl"
)


HARD_SOURCE = Path(
    "data/generated/hard_test_v1.jsonl"
)


SYSTEM_PROMPT = """
You are DeepLearning-Copilot.

Your task is deep learning experiment diagnosis.

You must:
- output valid JSON only;
- use project vocabulary;
- use canonical evidence codes;
- use canonical recommended action codes.
"""


TARGET = {

    "schema_alignment":1200,

    "evidence_grounding":1000,

    "recommendation_mapping":600,

    "directional_boundary":400,

    "priority_composition":400,

    "invariance_distractor":400,
}



def load_jsonl(path):

    data=[]

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



def make_id(category, index):

    value = (
        f"{category}_{index}"
    )

    return hashlib.md5(
        value.encode()
    ).hexdigest()[:12]



def build_item(
    sample,
    category,
    index,
):

    return {

        "id":
            make_id(
                category,
                index
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
                    sample["prompt"]
            },

            {
                "role":
                    "assistant",

                "content":
                    json.dumps(
                        sample["ground_truth"],
                        ensure_ascii=False,
                    )
            }
        ]
    }



def main():

    random.seed(2026)


    standard = load_jsonl(
        SOURCE
    )


    hard = load_jsonl(
        HARD_SOURCE
    )


    diagnosis = [
        x for x in standard
        if x["task_type"]
        ==
        "experiment_diagnosis"
    ]


    output=[]


    # 普通对齐数据

    for category,count in {

        "schema_alignment":1200,

        "evidence_grounding":1000,

        "recommendation_mapping":600,

    }.items():


        for i in range(count):

            sample=random.choice(
                diagnosis
            )

            output.append(
                build_item(
                    sample,
                    category,
                    i,
                )
            )



    # Hard Reasoning

    for category,count in {

        "directional_boundary":400,

        "priority_composition":400,

        "invariance_distractor":400,

    }.items():


        candidates=[

            x for x in hard

            if x[
                "test_property_type"
            ]
            ==
            category

        ]


        if not candidates:

            raise ValueError(
                f"No hard samples for {category}"
            )


        for i in range(count):

            sample=random.choice(
                candidates
            )

            output.append(
                build_item(
                    sample,
                    category,
                    i,
                )
            )


    random.shuffle(
        output
    )


    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        OUTPUT,
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
        "SFT V2 GENERATION PASSED"
    )

    print(
        "Samples:",
        len(output)
    )


    print(
        "Distribution:"
    )


    for k,v in TARGET.items():

        print(
            k,
            v
        )



if __name__=="__main__":
    main()
