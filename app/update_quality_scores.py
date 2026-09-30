import csv
from pathlib import Path


INPUT_FILE = Path("results/quality_evaluation.csv")
BACKUP_FILE = Path("results/quality_evaluation_backup.csv")


SCORE_COLUMNS = [
    "correctness_score",
    "instruction_adherence_score",
    "completeness_score",
]

QUALITY_COLUMN = "quality_score"


def parse_score(value, column_name, row_number):
    """
    Convert a CSV score to an integer.

    Blank values are allowed because some rows
    have not been evaluated yet.
    """

    value = value.strip()

    if value == "":
        return None

    try:
        score = int(value)
    except ValueError:
        raise ValueError(
            f"Invalid value '{value}' in "
            f"{column_name} at CSV row {row_number}. "
            "Expected 0, 1, or 2."
        )

    if score not in (0, 1, 2):
        raise ValueError(
            f"Invalid score {score} in "
            f"{column_name} at CSV row {row_number}. "
            "Scores must be 0, 1, or 2."
        )

    return score


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_FILE}"
        )

    # --------------------------------------------------
    # Load existing evaluation file
    # --------------------------------------------------

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "The CSV file has no header."
            )

        fieldnames = reader.fieldnames

        for column in SCORE_COLUMNS + [
            QUALITY_COLUMN
        ]:

            if column not in fieldnames:
                raise ValueError(
                    f"Missing required column: {column}"
                )

        rows = list(reader)

    # --------------------------------------------------
    # Create backup
    # --------------------------------------------------

    with open(
        BACKUP_FILE,
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

    # --------------------------------------------------
    # Calculate quality scores
    # --------------------------------------------------

    completed = 0
    incomplete = 0

    for csv_index, row in enumerate(
        rows,
        start=2,  # header is row 1
    ):

        scores = []

        for column in SCORE_COLUMNS:

            score = parse_score(
                row[column],
                column,
                csv_index,
            )

            scores.append(score)

        # ----------------------------------------------
        # Only calculate when all 3 scores exist
        # ----------------------------------------------

        if all(
            score is not None
            for score in scores
        ):

            quality_score = (
                sum(scores) / 3
            )

            row[QUALITY_COLUMN] = (
                f"{quality_score:.3f}"
            )

            completed += 1

        else:

            row[QUALITY_COLUMN] = ""

            incomplete += 1

    # --------------------------------------------------
    # Write updated CSV
    # --------------------------------------------------

    with open(
        INPUT_FILE,
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

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("=" * 60)
    print("QUALITY SCORE UPDATE")
    print("=" * 60)

    print(
        f"Total rows: {len(rows)}"
    )

    print(
        f"Quality scores calculated: "
        f"{completed}"
    )

    print(
        f"Rows still incomplete: "
        f"{incomplete}"
    )

    print(
        f"Updated file: {INPUT_FILE}"
    )

    print(
        f"Backup file: {BACKUP_FILE}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()