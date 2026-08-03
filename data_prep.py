"""
Loads the Financial PhraseBank dataset (Malo et al., 2014) from the
Hugging Face Hub, cleans it, and produces train/val/test splits.

NOTE ON SOURCE: the original `takala/financial_phrasebank` repo ships as a
Python "dataset loading script", which recent versions of the `datasets`
library refuse to run (the trust_remote_code escape hatch for scripts was
removed for security reasons — see https://github.com/huggingface/datasets
release notes). We instead load a pre-converted Parquet mirror,
`FinanceMTEB/financial_phrasebank`, which contains the same underlying
sentences restricted to the subset where all annotators agreed on the
label (the "AllAgree" subset, ~2,264 sentences) — the cleanest-label
version of Financial PhraseBank. If you specifically need the larger,
lower-agreement subsets (50/66/75agree), you'd need to source the
original .txt files from the dataset's data card and parse them manually.

Usage:
    python modeling/data_prep.py
"""

import argparse
import os

from datasets import load_dataset, DatasetDict, ClassLabel

LABEL_NAMES = ["negative", "neutral", "positive"]
HF_DATASET_ID = "FinanceMTEB/financial_phrasebank"


def load_and_split(seed: int = 42) -> DatasetDict:
    """Load Financial PhraseBank (AllAgree subset, via Parquet mirror) and
    create train/val/test splits.

    The mirror ships its own train/test split, but we recombine and
    re-split ourselves so we get a proper train/val/test three-way split
    with consistent proportions.
    """
    raw = load_dataset(HF_DATASET_ID)

    # Combine whatever splits the mirror provides into one pool, then
    # re-split ourselves for a clean train/val/test partition.
    from datasets import concatenate_datasets

    ds = concatenate_datasets([raw[split] for split in raw.keys()])

    if not isinstance(ds.features["label"], ClassLabel):
        ds = ds.cast_column("label", ClassLabel(names=LABEL_NAMES))

    # 80/10/10 split
    split1 = ds.train_test_split(test_size=0.2, seed=seed, stratify_by_column="label")
    split2 = split1["test"].train_test_split(test_size=0.5, seed=seed, stratify_by_column="label")

    return DatasetDict(
        train=split1["train"],
        validation=split2["train"],
        test=split2["test"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/financial_phrasebank")
    args, _unknown = parser.parse_known_args()

    dsd = load_and_split()

    os.makedirs(args.out_dir, exist_ok=True)
    dsd.save_to_disk(args.out_dir)

    print("Split sizes:")
    for split_name, split in dsd.items():
        print(f"  {split_name}: {len(split)}")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
