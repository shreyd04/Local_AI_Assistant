import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Literal

from ollama import Client
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results"
    / "quality_evaluation.csv"
)

DEFAULT_OUTPUT = DEFAULT_INPUT

DEFAULT_BACKUP = (
    PROJECT_ROOT
    / "results"
    / "quality_evaluation_before_auto_scoring.csv"
)

RUBRIC_FILE = (
    PROJECT_ROOT
    / "benchmarks"
    / "quality_rubric.json"
)

OLLAMA_HOST = "http://localhost:11434"

# Local model used as evaluation judge.
JUDGE_MODEL = "mistral:7b-instruct-v0.3-q5_K_M"

client = Client(host=OLLAMA_HOST)


class QualityJudgment(BaseModel):
    correctness_score: Literal[0, 1, 2]
    instruction_adherence_score: Literal[0, 1, 2]
    completeness_score: Literal[0, 1, 2]
    notes: str = Field(default="", max_length=1000)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated rubric-based LLM quality evaluation."
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input CSV containing model responses.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV. Defaults to input CSV.",
    )

    parser.add_argument(
        "--backup",
        default=None,
        help="Backup CSV created before scoring.",
    )

    return parser.parse_args()


def load_rubric():
    with open(
        RUBRIC_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def has_score(value):
    return (
        value is not None
        and str(value).strip() != ""
    )


def evaluate_response(
    model_name,
    category,
    prompt,
    output,
    rubric,
):
    dimension_rules = rubric["dimensions"]

    category_guidance = rubric[
        "category_guidance"
    ][category]

    judge_schema = (
        QualityJudgment.model_json_schema()
    )

    system_prompt = """
You are a strict evaluation judge for a local LLM benchmark.

Evaluate the model response against the USER PROMPT and
the provided evaluation rubric.

Do not compare the response with other models.
Do not reward verbosity.
Do not punish an answer merely for using different wording.
Judge only whether the response satisfies the stated task.

Scores:
0 = fails the criterion or contains a material error.
1 = partially satisfies the criterion.
2 = fully satisfies the criterion.

Be conservative and evidence-based.

Return ONLY JSON matching the provided schema.
""".strip()

    user_prompt = f"""
CATEGORY:
{category}

USER PROMPT:
{prompt}

MODEL:
{model_name}

MODEL RESPONSE:
{output}

RUBRIC FOR THIS CATEGORY:
{json.dumps(category_guidance, indent=2)}

DIMENSION DEFINITIONS:

Correctness:
{dimension_rules["correctness"][category]}

Instruction adherence:
{dimension_rules["instruction_adherence"][category]}

Completeness:
{dimension_rules["completeness"][category]}

Return JSON matching this schema:

{json.dumps(judge_schema, indent=2)}
""".strip()

    response = client.chat(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        format=judge_schema,
        options={
            "temperature": 0.0,
        },
        stream=False,
    )

    raw = response.message.content

    return QualityJudgment.model_validate_json(raw)


def calculate_quality_score(row):
    scores = [
        int(row["correctness_score"]),
        int(row["instruction_adherence_score"]),
        int(row["completeness_score"]),
    ]

    return sum(scores) / 3


def ensure_score_columns(fieldnames, rows):
    score_columns = [
        "correctness_score",
        "instruction_adherence_score",
        "completeness_score",
        "quality_score",
        "notes",
    ]

    for column in score_columns:
        if column not in fieldnames:
            fieldnames.append(column)

            for row in rows:
                row[column] = ""

    return fieldnames, rows


def create_summary(rows, output_file):
    scored_rows = []

    for row in rows:
        if all(
            has_score(row.get(column, ""))
            for column in [
                "correctness_score",
                "instruction_adherence_score",
                "completeness_score",
            ]
        ):
            scored_rows.append(row)

    if not scored_rows:
        return

    model_values = sorted(
        set(row["model"] for row in scored_rows)
    )

    summary_file = (
        output_file.parent
        / f"{output_file.stem}_summary.csv"
    )

    category_file = (
        output_file.parent
        / f"{output_file.stem}_category_summary.csv"
    )

    summary_rows = []

    for model in model_values:
        model_rows = [
            row
            for row in scored_rows
            if row["model"] == model
        ]

        def mean(column):
            values = [
                float(row[column])
                for row in model_rows
            ]
            return sum(values) / len(values)

        quality_mean = mean("quality_score")

        summary_rows.append(
            {
                "model": model,
                "scored_prompts": len(model_rows),
                "mean_quality_score": quality_mean,
                "quality_percentage": (
                    quality_mean / 2.0
                ) * 100,
                "correctness_mean": mean(
                    "correctness_score"
                ),
                "instruction_adherence_mean": mean(
                    "instruction_adherence_score"
                ),
                "completeness_mean": mean(
                    "completeness_score"
                ),
            }
        )

    with open(
        summary_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    category_rows = []

    categories = sorted(
        set(row["category"] for row in scored_rows)
    )

    for category in categories:

        category_data = [
            row
            for row in scored_rows
            if row["category"] == category
        ]

        def category_mean(column):
            values = [
                float(row[column])
                for row in category_data
            ]
            return sum(values) / len(values)

        category_rows.append(
            {
                "category": category,
                "scored_prompts": len(category_data),
                "mean_quality_score": category_mean(
                    "quality_score"
                ),
                "correctness_mean": category_mean(
                    "correctness_score"
                ),
                "instruction_adherence_mean": category_mean(
                    "instruction_adherence_score"
                ),
                "completeness_mean": category_mean(
                    "completeness_score"
                ),
            }
        )

    with open(
        category_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=category_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(category_rows)

    return summary_file, category_file


def main():

    args = parse_args()

    input_file = Path(args.input)

    output_file = (
        Path(args.output)
        if args.output
        else input_file
    )

    backup_file = (
        Path(args.backup)
        if args.backup
        else DEFAULT_BACKUP
    )

    if not input_file.exists():
        raise FileNotFoundError(
            f"Missing input file: {input_file}"
        )

    if not RUBRIC_FILE.exists():
        raise FileNotFoundError(
            f"Missing rubric: {RUBRIC_FILE}"
        )

    rubric = load_rubric()

    with open(
        input_file,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "CSV has no header."
            )

        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    required_base_columns = [
        "model",
        "prompt_id",
        "category",
        "prompt",
        "output",
    ]

    for column in required_base_columns:
        if column not in fieldnames:
            raise ValueError(
                f"Missing CSV column: {column}"
            )

    fieldnames, rows = ensure_score_columns(
        fieldnames,
        rows,
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Do not overwrite the source before creating its backup.
    if input_file.resolve() == output_file.resolve():

        shutil.copy2(
            input_file,
            backup_file,
        )

    print("=" * 70)
    print("AUTOMATED QUALITY SCORING")
    print("=" * 70)

    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Total rows: {len(rows)}")
    print("=" * 70)

    evaluated = 0
    already_complete = 0
    failures = 0

    score_columns = [
        "correctness_score",
        "instruction_adherence_score",
        "completeness_score",
    ]

    for index, row in enumerate(
        rows,
        start=1,
    ):

        model_name = row["model"]
        prompt_id = row["prompt_id"]
        category = row["category"]

        prompt = row["prompt"]
        output = row["output"]

        all_scores_exist = all(
            has_score(row[column])
            for column in score_columns
        )

        if all_scores_exist:

            row["quality_score"] = (
                f"{calculate_quality_score(row):.3f}"
            )

            already_complete += 1

            print(
                f"[{index}/{len(rows)}] "
                f"{model_name} | "
                f"{prompt_id} | "
                f"already scored"
            )

            continue

        if not output.strip():

            failures += 1

            print(
                f"[{index}/{len(rows)}] "
                f"{model_name} | "
                f"{prompt_id} | "
                f"FAILED: empty output"
            )

            continue

        print(
            f"[{index}/{len(rows)}] "
            f"{model_name} | "
            f"{prompt_id} | evaluating..."
        )

        try:

            judgment = evaluate_response(
                model_name=model_name,
                category=category,
                prompt=prompt,
                output=output,
                rubric=rubric,
            )

            # Fill only blank fields.
            if not has_score(
                row["correctness_score"]
            ):
                row["correctness_score"] = str(
                    judgment.correctness_score
                )

            if not has_score(
                row["instruction_adherence_score"]
            ):
                row["instruction_adherence_score"] = str(
                    judgment.instruction_adherence_score
                )

            if not has_score(
                row["completeness_score"]
            ):
                row["completeness_score"] = str(
                    judgment.completeness_score
                )

            row["quality_score"] = (
                f"{calculate_quality_score(row):.3f}"
            )

            if not row["notes"].strip():
                row["notes"] = (
                    "[Automated judge] "
                    + judgment.notes
                )

            evaluated += 1

            print(
                "    Scores: "
                f"C={row['correctness_score']} "
                f"I={row['instruction_adherence_score']} "
                f"Co={row['completeness_score']} "
                f"Q={row['quality_score']}"
            )

        except Exception as exc:

            failures += 1

            print(
                f"    FAILED: {exc}"
            )

    # Write output.
    with open(
        output_file,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    summary_result = create_summary(
        rows,
        output_file,
    )

    print()
    print("=" * 70)
    print("AUTOMATED QUALITY SCORING COMPLETE")
    print("=" * 70)

    print(f"Total rows: {len(rows)}")
    print(f"Newly evaluated: {evaluated}")
    print(f"Already scored: {already_complete}")
    print(f"Failures: {failures}")
    print(f"Output file: {output_file}")

    if input_file.resolve() == output_file.resolve():
        print(f"Backup file: {backup_file}")

    if summary_result:
        summary_file, category_file = summary_result

        print(
            f"Model summary: {summary_file}"
        )

        print(
            f"Category summary: {category_file}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()