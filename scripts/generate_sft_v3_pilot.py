import json
import random
import hashlib
from pathlib import Path


OUTPUT = Path(
    "data/generated/sft_v3_pilot.jsonl"
)


HARD_SOURCE = Path(
    "data/generated/hard_test_v1.jsonl"
)


SYSTEM_PROMPT = """
You are DeepLearning-Copilot.

You must diagnose deep learning experiments.

Rules:
- Output JSON only.
- Use canonical project vocabulary.
- Follow evidence codes.
- Follow recommended action codes.
- Resolve conflicting evidence according to project priority rules.
"""


TARGET = {

    "priority_composition": 2000,

    "directional_boundary": 1200,

    "output_completion": 800,

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

    value=f"{category}_{index}"

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


    hard = load_jsonl(
        HARD_SOURCE
    )


    output=[]


    categories=[]

    for k,v in TARGET.items():

        categories.extend(
            [k]*v
        )


    random.shuffle(
        categories
    )


    for idx,category in enumerate(
        categories
    ):


        if category == "priority_composition":

            candidates=[

                x for x in hard

                if "HT_PC" in
                x.get(
                    "hard_test_family_id",
                    ""
                )

            ]


        elif category == "directional_boundary":

            candidates=[

                x for x in hard

                if "HT_DIR" in
                x.get(
                    "hard_test_family_id",
                    ""
                )

            ]


        else:

            candidates=hard



        if not candidates:

            raise ValueError(
                f"No candidates for {category}"
            )


        sample=random.choice(
            candidates
        )


        output.append(
            build_item(
                sample,
                category,
                idx,
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
        "SFT V3 GENERATION PASSED"
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
