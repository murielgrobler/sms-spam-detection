import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("sms_spam_collection_data/SMSSpamCollection"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/sms_spam.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input
    output_path = args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(
        input_path,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8",
        encoding_errors="replace",
    )

    df["label"] = df["label"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()

    df = df[(df["label"] != "") & (df["text"] != "")].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Wrote: {output_path}")
    print("Rows:", len(df))
    print("Label counts:\n", df["label"].value_counts(dropna=False))
    print("Duplicate texts:", int(df["text"].duplicated().sum()))


if __name__ == "__main__":
    main()
