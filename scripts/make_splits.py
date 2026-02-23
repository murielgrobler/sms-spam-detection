import argparse
import json
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/sms_spam.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/splits"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    return parser.parse_args()


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def main() -> None:
    args = parse_args()

    if abs((args.train + args.val + args.test) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")

    df = pd.read_csv(args.input)
    df["label"] = df["label"].astype(str)
    df["text"] = df["text"].astype(str)

    df["text_id"] = df["text"].map(_stable_hash)

    group = (
        df.groupby("text_id", as_index=False)
        .agg(label_list=("label", lambda s: sorted(set(s))))
        .reset_index(drop=True)
    )

    conflicts = group[group["label_list"].map(len) != 1]
    dropped_conflict_groups = int(len(conflicts))
    dropped_conflict_rows = 0
    if dropped_conflict_groups:
        conflict_ids = set(conflicts["text_id"].tolist())
        dropped_conflict_rows = int(df["text_id"].isin(conflict_ids).sum())
        df = df[~df["text_id"].isin(conflict_ids)].reset_index(drop=True)
        group = group[~group["text_id"].isin(conflict_ids)].reset_index(drop=True)
        print(
            "Dropping duplicated texts with conflicting labels. "
            f"Groups dropped: {dropped_conflict_groups}. Rows dropped: {dropped_conflict_rows}."
        )

    group["label"] = group["label_list"].map(lambda x: x[0])

    group_ids = group["text_id"].tolist()
    group_labels = group["label"].tolist()

    train_ids, temp_ids, train_y, temp_y = train_test_split(
        group_ids,
        group_labels,
        test_size=(1.0 - args.train),
        random_state=args.seed,
        stratify=group_labels,
    )

    val_frac_of_temp = args.val / (args.val + args.test)
    val_ids, test_ids, _, _ = train_test_split(
        temp_ids,
        temp_y,
        test_size=(1.0 - val_frac_of_temp),
        random_state=args.seed,
        stratify=temp_y,
    )

    split_map = {
        "train": set(train_ids),
        "val": set(val_ids),
        "test": set(test_ids),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, ids in split_map.items():
        out_path = args.out_dir / f"{split_name}.csv"
        df[df["text_id"].isin(ids)].drop(columns=["text_id"]).to_csv(out_path, index=False)

    summary = {
        "input": str(args.input),
        "seed": args.seed,
        "fractions": {"train": args.train, "val": args.val, "test": args.test},
        "rows_total": int(len(df)),
        "unique_texts": int(df["text"].nunique()),
        "unique_text_ids": int(df["text_id"].nunique()),
        "dropped_conflicting_label_groups": dropped_conflict_groups,
        "dropped_conflicting_label_rows": dropped_conflict_rows,
        "splits": {},
    }

    for split_name in ["train", "val", "test"]:
        sdf = pd.read_csv(args.out_dir / f"{split_name}.csv")
        summary["splits"][split_name] = {
            "rows": int(len(sdf)),
            "label_counts": sdf["label"].value_counts().to_dict(),
            "duplicate_texts": int(sdf["text"].duplicated().sum()),
        }

    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote splits to: {args.out_dir}")
    print(json.dumps(summary["splits"], indent=2))


if __name__ == "__main__":
    main()
