import csv
import json
import shutil
from pathlib import Path
from typing import Literal

from ollama import Client
from pydantic import BaseModel, Field


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CSV_FILE = Path(
    "results/quality_evaluation.csv"
)

BACKUP_FILE = Path(
    "results/quality_evaluation_before_auto_scoring.csv"
)

RUBRIC_FILE = Path(
    "benchmarks/quality_rubric.json"
)

OLLAMA_HOST = "http://localhost:11434"

# Local model used as the evaluation judge.
JUDGE_MODEL = "mistral:7b-instruct-v0.3-q5_K_M"

client = Client(
    host=OLLAMA_HOST
)


# --------------------------------------------------
# Pydantic schema for judge output
# --------------------------------------------------

class QualityJudgment(BaseModel):

    correctness_score: Literal[
        0,
        1,
        2,
    ]

    instruction_adherence_score: Literal[
        0,
        1,
        2,
    ]

    completeness_score: Literal[
        0,
        1,
        2,
    ]

    notes: str = Field(
        default="",
        max_length=1000,
    )


# --------------------------------------------------
# Load rubric
# --------------------------------------------------

def load_rubric():

    with open(
        RUBRIC_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# --------------------------------------------------
# Check whether a score is present
# --------------------------------------------------

def has_score(value):

    return (
        value is not None
        and str(value).strip() != ""
    )


# --------------------------------------------------
# Ask local model to judge one response
# --------------------------------------------------

def evaluate_response(
    model_name,
    category,
    prompt,
    output,
    rubric,
):

    dimension_rules = rubric[
        "dimensions"
    ]

    category_guidance = rubric[
        "category_guidance"
    ][category]

    judge_schema = (
        QualityJudgment
        .model_json_schema()
    )

    system_prompt = """
You are a strict evaluation judge for a local LLM benchmark.

Evaluate the model response against the USER PROMPT and
the provided evaluation rubric.

Do not compare the response with other models.
Do not reward verbosity.
Do not punish an answer merely for using different wording.
Judge only whether the response satisfies the stated task.

Return ONLY JSON matching the provided schema.

Scores:
0 = fails the criterion or contains a material error.
1 = partially satisfies the criterion.
2 = fully satisfies the criterion.

Be conservative and evidence-based.
"""

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
"""

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

    judgment = (
        QualityJudgment
        .model_validate_json(raw)
    )

    return judgment


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    if not CSV_FILE.exists():

        raise FileNotFoundError(
            f"Missing file: {CSV_FILE}"
        )

    if not RUBRIC_FILE.exists():

        raise FileNotFoundError(
            f"Missing rubric: {RUBRIC_FILE}"
        )

    rubric = load_rubric()

    # ----------------------------------------------
    # Read CSV
    # ----------------------------------------------

    with open(
        CSV_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "CSV has no header."
            )

        fieldnames = list(
            reader.fieldnames
        )

        rows = list(reader)

    # ----------------------------------------------
    # Validate required columns
    # ----------------------------------------------

    required_columns = [
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

    for column in required_columns:

        if column not in fieldnames:

            raise ValueError(
                f"Missing CSV column: {column}"
            )

    # ----------------------------------------------
    # Create backup BEFORE modification
    # ----------------------------------------------

    shutil.copy2(
        CSV_FILE,
        BACKUP_FILE,
    )

    print("=" * 70)
    print("AUTOMATED QUALITY SCORING")
    print("=" * 70)

    print(
        f"Judge model: {JUDGE_MODEL}"
    )

    print(
        f"Total rows: {len(rows)}"
    )

    print("=" * 70)

    evaluated = 0
    already_complete = 0
    failures = 0

    # ----------------------------------------------
    # Process every row
    # ----------------------------------------------

    for index, row in enumerate(
    rows,
    start=1,
    ):

        model_name = row[
            "model"
        ]
  
        prompt_id = row[
            "prompt_id"
        ]

        category = row[
            "category"
        ]

        prompt = row[
            "prompt"
        ]

        output = row[
            "output"
        ]

        # ------------------------------------------
        # Check existing scores
        # ------------------------------------------

        score_columns = [
            "correctness_score",
            "instruction_adherence_score",
            "completeness_score",
        ]

        all_scores_exist = all(
            has_score(
                row[column]
            )
            for column in score_columns
        )

        if all_scores_exist:

            # Recalculate quality score
            scores = [
                int(
                    row[column]
                )
                for column in score_columns
            ]

            row[
                "quality_score"
            ] = f"{sum(scores) / 3:.3f}"

            already_complete += 1

            print(
                f"[{index}/{len(rows)}] "
                f"{model_name} | "
                f"{prompt_id} | "
                f"already scored"
            )

            continue

        # ------------------------------------------
        # Validate output exists
        # ------------------------------------------

        if not output.strip():

            failures += 1

            print(
                f"[{index}/{len(rows)}] "
                f"{model_name} | "
                f"{prompt_id} | "
                f"FAILED: empty output"
            )

            continue

        # ------------------------------------------
        # Judge response
        # ------------------------------------------

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

            # --------------------------------------
            # Preserve existing manual scores.
            # Fill only blanks.
            # --------------------------------------

            if not has_score(
                row[
                    "correctness_score"
                ]
            ):

                row[
                    "correctness_score"
                ] = str(
                    judgment.correctness_score
                )

            if not has_score(
                row[
                    "instruction_adherence_score"
                ]
            ):

                row[
                    "instruction_adherence_score"
                ] = str(
                    judgment.instruction_adherence_score
                )

            if not has_score(
                row[
                    "completeness_score"
                ]
            ):

                row[
                    "completeness_score"
                ] = str(
                    judgment.completeness_score
                )

            # --------------------------------------
            # Calculate quality score
            # --------------------------------------

            scores = [
                int(
                    row[
                        "correctness_score"
                    ]
                ),
                int(
                    row[
                        "instruction_adherence_score"
                    ]
                ),
                int(
                    row[
                        "completeness_score"
                    ]
                ),
            ]

            row[
                "quality_score"
            ] = f"{sum(scores) / 3:.3f}"

            # --------------------------------------
            # Add automated note only if notes empty
            # --------------------------------------

            if not row["notes"].strip():

                row[
                    "notes"
                ] = (
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

    # ----------------------------------------------
    # Write updated CSV
    # ----------------------------------------------

    with open(
        CSV_FILE,
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

    # ----------------------------------------------
    # Summary
    # ----------------------------------------------

    print()
    print("=" * 70)
    print("AUTOMATED QUALITY SCORING COMPLETE")
    print("=" * 70)

    print(
        f"Total rows: {len(rows)}"
    )

    print(
        f"Newly evaluated: {evaluated}"
    )

    print(
        f"Already scored: {already_complete}"
    )

    print(
        f"Failures: {failures}"
    )

    print(
        f"Updated file: {CSV_FILE}"
    )

    print(
        f"Backup file: {BACKUP_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()