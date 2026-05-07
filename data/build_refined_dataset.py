"""
build_refined_dataset.py

Reproduces the Refined Dataset described in the paper:
"Re-labeling Approach for Spanish-English Code-switching Sentiment Analysis:
 Impact of Data Quality Improvement" (KSC 2025, Honorable Mention)

This script loads the original LINCE SA (sa_spaeng) split from Hugging Face,
applies the human-verified label refinements in label_mapping.json, and writes
the resulting refined dataset (5,567 samples) to refined_dataset.json.

Usage:
    python build_refined_dataset.py

Output:
    refined_dataset.json — the refined dataset, ready for downstream training
"""

import json
import os
from collections import Counter

# The original LINCE SA dataset is loaded via Hugging Face datasets.
# Install with: pip install datasets
from datasets import load_dataset


# Splits used to build the refined dataset.
# The test split is excluded because LINCE SA does not provide gold
# sentiment labels for the test split.
SPLITS_TO_USE = ["train", "validation"]

# LINCE SA returns the sentiment label as a string from this set.
VALID_LABELS = {"positive", "neutral", "negative"}


def load_label_mapping(path="label_mapping.json"):
    """Load the human-verified label refinements."""
    mapping_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), path
    )
    with open(mapping_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw["label_mapping"]


def is_code_switched(lid_labels, min_tokens_per_lang=2):
    """
    A sample is treated as genuine Spanish-English code-switching when both
    lang1 (English) and lang2 (Spanish) have at least min_tokens_per_lang
    tokens.
    """
    counts = Counter(lid_labels)
    return (
        counts.get("lang1", 0) >= min_tokens_per_lang
        and counts.get("lang2", 0) >= min_tokens_per_lang
    )


def build_refined_dataset():
    """
    1. Load LINCE SA (sa_spaeng) from Hugging Face
    2. Filter for genuine code-switching samples (both languages, >=2 tokens each)
    3. Assign global sample_ids that align with label_mapping.json
    4. Apply human-verified label refinements
    5. Drop samples whose sample_id appears in more than one split
    6. Write refined_dataset.json
    """
    print("Loading label refinements from label_mapping.json ...")
    label_mapping = load_label_mapping()
    print(f"  {len(label_mapping)} refinements loaded\n")

    print("Loading LINCE SA from Hugging Face ...")
    dataset = load_dataset(
        "lince-benchmark/lince", "sa_spaeng",
        trust_remote_code=True,
    )

    # Enumerate samples across train + validation, assigning global indices.
    # The global index becomes the sample_id (sample_<idx>).
    # This ID scheme is what label_mapping.json keys reference.
    all_samples = []
    global_idx = 0

    for split in SPLITS_TO_USE:
        split_data = dataset[split]
        print(f"  {split}: {len(split_data)} samples")

        for row in split_data:
            sample_id = f"sample_{global_idx}"
            global_idx += 1

            sentiment = row["sa"]
            if sentiment not in VALID_LABELS:
                continue
            if not is_code_switched(row["lid"]):
                continue

            # Apply a refinement when one is recorded for this sample_id.
            original_sentiment = sentiment
            if sample_id in label_mapping:
                ref = label_mapping[sample_id]
                # Sanity guard: only apply when the source label still
                # matches the recorded original.
                if ref["original"] == sentiment:
                    sentiment = ref["corrected"]

            all_samples.append({
                "id": sample_id,
                "text": " ".join(row["words"]),
                "tokens": row["words"],
                "lid_labels": row["lid"],
                "sentiment": sentiment,
                "original_sentiment": original_sentiment,
                "label_refined": sentiment != original_sentiment,
            })

    print(f"\nCode-switched samples collected: {len(all_samples)}")

    # Drop samples whose sample_id is shared across splits, to avoid
    # ambiguous labels in the final set.
    id_counts = Counter(s["id"] for s in all_samples)
    duplicate_ids = {sid for sid, c in id_counts.items() if c > 1}
    print(f"Cross-split duplicate IDs removed: {len(duplicate_ids)}")

    refined = [s for s in all_samples if s["id"] not in duplicate_ids]
    print(f"Refined dataset size: {len(refined)}")

    dist = Counter(s["sentiment"] for s in refined)
    print("\nLabel distribution:")
    for label, count in sorted(dist.items()):
        print(f"  {label}: {count} ({count / len(refined) * 100:.1f}%)")

    refinements_applied = sum(1 for s in refined if s["label_refined"])
    print(f"\nLabel refinements applied: {refinements_applied}")

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "refined_dataset.json",
    )
    output = {
        "metadata": {
            "dataset": "LINCE SA Refined (Spanish-English Sentiment)",
            "source": "lince-benchmark/lince [sa_spaeng] via Hugging Face",
            "paper": (
                "Re-labeling Approach for Spanish-English Code-switching "
                "Sentiment Analysis (KSC 2025)"
            ),
            "total_samples": len(refined),
            "label_refinements_applied": refinements_applied,
            "label_distribution": dict(dist),
        },
        "data": refined,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {output_path}")
    print("Done.")


if __name__ == "__main__":
    build_refined_dataset()
