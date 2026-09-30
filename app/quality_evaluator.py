import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


RESULTS_DIR = Path("results")
PROMPT_FILE = Path("benchmarks/prompts.json")
RUBRIC_FILE = Path("benchmarks/quality_rubric.json")

EVALUATION_COLUMNS = [
    "model",
    "prompt_id",
    "category",
    "prompt",
    "output",
    "correctness_score",
    "instruction_adherence_score",
    "completeness_score",
    "quality_score",
    "notes",
]


def load_rubric():
    with open(
        RUBRIC_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_prompt_metadata():
    with open(
        PROMPT_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        prompts = json.load(file)

    return {
        item["id"]: item
        for item in prompts
    }


def find_raw_result_files():
    return sorted(
        RESULTS_DIR.glob("*_raw.csv")
    )


def load_raw_results():
    prompt_metadata = load_prompt_metadata()

    rows = []

    for filepath in find_raw_result_files():

        with open(
            filepath,
            "r",
            encoding="utf-8",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                prompt_id = row["prompt_id"]

                metadata = prompt_metadata.get(
                    prompt_id
                )

                if metadata is None:
                    raise ValueError(
                        f"Prompt ID not found: "
                        f"{prompt_id}"
                    )

                rows.append(
                    {
                        "model": row["model"],
                        "prompt_id": prompt_id,
                        "category": metadata[
                            "category"
                        ],
                        "prompt": metadata[
                            "prompt"
                        ],
                        "output": row[
                            "output"
                        ],
                        "correctness_score": "",
                        "instruction_adherence_score": "",
                        "completeness_score": "",
                        "quality_score": "",
                        "notes": "",
                    }
                )

    return rows


def load_existing_scores(
    evaluation_file
):
    if not evaluation_file.exists():
        return {}

    saved = {}

    with open(
        evaluation_file,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            key = (
                row["model"],
                row["prompt_id"],
            )

            saved[key] = row

    return saved


def merge_existing_scores(
    rows,
    existing,
):
    for row in rows:

        key = (
            row["model"],
            row["prompt_id"],
        )

        previous = existing.get(key)

        if previous is None:
            continue

        row[
            "correctness_score"
        ] = previous.get(
            "correctness_score",
            "",
        )

        row[
            "instruction_adherence_score"
        ] = previous.get(
            "instruction_adherence_score",
            "",
        )

        row[
            "completeness_score"
        ] = previous.get(
            "completeness_score",
            "",
        )

        row["quality_score"] = previous.get(
            "quality_score",
            "",
        )

        row["notes"] = previous.get(
            "notes",
            "",
        )


def save_evaluation(
    rows,
    filepath,
):

    filepath.parent.mkdir(
        exist_ok=True
    )

    with open(
        filepath,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=EVALUATION_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(rows)


def validate_score(value):
    if value == "":
        return None

    score = int(value)

    if score not in (0, 1, 2):
        raise ValueError(
            "Scores must be 0, 1, or 2."
        )

    return score


def calculate_quality_score(row):

    scores = [
        validate_score(
            row["correctness_score"]
        ),
        validate_score(
            row["instruction_adherence_score"]
        ),
        validate_score(
            row["completeness_score"]
        ),
    ]

    if any(
        score is None
        for score in scores
    ):
        return ""

    return round(
        sum(scores) / len(scores),
        3,
    )


def create_summary(rows):

    model_stats = defaultdict(
        lambda: {
            "scores": [],
            "correctness": [],
            "instruction": [],
            "completeness": [],
        }
    )

    category_stats = defaultdict(
        lambda: {
            "scores": [],
        }
    )

    for row in rows:

        quality_score = calculate_quality_score(
            row
        )

        if quality_score == "":
            continue

        model = row["model"]
        category = row["category"]

        model_stats[model][
            "scores"
        ].append(quality_score)

        model_stats[model][
            "correctness"
        ].append(
            int(row["correctness_score"])
        )

        model_stats[model][
            "instruction"
        ].append(
            int(
                row[
                    "instruction_adherence_score"
                ]
            )
        )

        model_stats[model][
            "completeness"
        ].append(
            int(
                row[
                    "completeness_score"
                ]
            )
        )

        category_stats[
            (
                model,
                category,
            )
        ]["scores"].append(
            quality_score
        )

    model_summary = []

    for model, stats in sorted(
        model_stats.items()
    ):

        model_summary.append(
            {
                "model": model,
                "scored_prompts": len(
                    stats["scores"]
                ),
                "mean_quality_score": round(
                    sum(stats["scores"])
                    / len(stats["scores"]),
                    3,
                ),
                "quality_percentage": round(
                    (
                        sum(stats["scores"])
                        / len(stats["scores"])
                    )
                    / 2
                    * 100,
                    2,
                ),
                "mean_correctness": round(
                    sum(stats["correctness"])
                    / len(stats["correctness"]),
                    3,
                ),
                "mean_instruction_adherence": round(
                    sum(stats["instruction"])
                    / len(stats["instruction"]),
                    3,
                ),
                "mean_completeness": round(
                    sum(stats["completeness"])
                    / len(stats["completeness"]),
                    3,
                ),
            }
        )

    category_summary = []

    for (
        model,
        category,
    ), stats in sorted(
        category_stats.items()
    ):

        category_summary.append(
            {
                "model": model,
                "category": category,
                "scored_prompts": len(
                    stats["scores"]
                ),
                "mean_quality_score": round(
                    sum(stats["scores"])
                    / len(stats["scores"]),
                    3,
                ),
                "quality_percentage": round(
                    (
                        sum(stats["scores"])
                        / len(stats["scores"])
                    )
                    / 2
                    * 100,
                    2,
                ),
            }
        )

    return (
        model_summary,
        category_summary,
    )


def save_summary(
    model_summary,
    category_summary,
):

    model_file = (
        RESULTS_DIR /
        "quality_model_summary.csv"
    )

    category_file = (
        RESULTS_DIR /
        "quality_category_summary.csv"
    )

    with open(
        model_file,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        if model_summary:

            writer = csv.DictWriter(
                file,
                fieldnames=model_summary[0].keys(),
            )

            writer.writeheader()
            writer.writerows(model_summary)

    with open(
        category_file,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        if category_summary:

            writer = csv.DictWriter(
                file,
                fieldnames=category_summary[
                    0
                ].keys(),
            )

            writer.writeheader()
            writer.writerows(
                category_summary
            )

    return (
        model_file,
        category_file,
    )


def print_instructions(rubric):

    print()
    print("=" * 70)
    print("QUALITY EVALUATION SYSTEM")
    print("=" * 70)

    print(
        "\nScore each dimension from 0 to 2:"
    )

    print(
        "0 = fails criterion / material error"
    )

    print(
        "1 = partially satisfies criterion"
    )

    print(
        "2 = fully satisfies criterion"
    )

    print(
        "\nEach prompt receives three scores:"
    )

    print(
        "1. correctness_score"
    )

    print(
        "2. instruction_adherence_score"
    )

    print(
        "3. completeness_score"
    )

    print(
        "\nquality_score = mean of the three"
    )

    print(
        "\nUse the category-specific guidance "
        "from benchmarks/quality_rubric.json."
    )

    print("=" * 70)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help=(
            "Generate/refresh the evaluation "
            "CSV and preserve existing scores."
        ),
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help=(
            "Calculate summaries from completed "
            "quality scores."
        ),
    )

    args = parser.parse_args()

    if not args.evaluate and not args.summary:

        parser.error(
            "Use --evaluate or --summary."
        )

    rubric = load_rubric()

    evaluation_file = (
        RESULTS_DIR /
        "quality_evaluation.csv"
    )

    if args.evaluate:

        rows = load_raw_results()

        existing = load_existing_scores(
            evaluation_file
        )

        merge_existing_scores(
            rows,
            existing,
        )

        save_evaluation(
            rows,
            evaluation_file,
        )

        print_instructions(
            rubric
        )

        print()
        print(
            f"Evaluation rows created: "
            f"{len(rows)}"
        )

        print(
            f"File: {evaluation_file}"
        )

    if args.summary:

        rows = []

        with open(
            evaluation_file,
            "r",
            encoding="utf-8",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:
                rows.append(row)

        (
            model_summary,
            category_summary,
        ) = create_summary(rows)

        files = save_summary(
            model_summary,
            category_summary,
        )

        print()
        print("=" * 70)
        print("QUALITY SUMMARY")
        print("=" * 70)

        if not model_summary:

            print(
                "No completed quality scores yet."
            )

        else:

            for row in model_summary:

                print(
                    f"\nModel: {row['model']}"
                )

                print(
                    f"Scored prompts: "
                    f"{row['scored_prompts']}"
                )

                print(
                    f"Mean quality score: "
                    f"{row['mean_quality_score']:.3f}/2"
                )

                print(
                    f"Quality percentage: "
                    f"{row['quality_percentage']:.2f}%"
                )

                print(
                    f"Correctness: "
                    f"{row['mean_correctness']:.3f}/2"
                )

                print(
                    f"Instruction adherence: "
                    f"{row['mean_instruction_adherence']:.3f}/2"
                )

                print(
                    f"Completeness: "
                    f"{row['mean_completeness']:.3f}/2"
                )

        print()
        print(
            f"Model summary: {files[0]}"
        )

        print(
            f"Category summary: {files[1]}"
        )


if __name__ == "__main__":
    main()