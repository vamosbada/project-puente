"""
build_refined_dataset.py

Reproduces the Refined Dataset used in the paper:
"A Relabeling Approach for Spanish-English Code-Switching Sentiment Analysis:
 Impact Analysis of Data Quality Improvement" (KSC 2025)

The original LINCE SA dataset contains labeling errors (~17% error rate).
This script applies our human-verified label corrections (label_mapping.json)
to reconstruct the Refined Dataset (5,567 samples) from the original source.

Usage:
    python build_refined_dataset.py

Output:
    refined_dataset.json  — corrected dataset ready for model training
"""

import json
import os
from collections import Counter

# The original LINCE SA dataset is available on Hugging Face.
# Install with: pip install datasets
from datasets import load_dataset


# Label encoding used throughout the paper
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}

# Splits used to build the preprocessed dataset (test split excluded)
SPLITS_TO_USE = ["train", "validation"]


def load_label_mapping(path="label_mapping.json"):
    """Load the human-verified label corrections."""
    mapping_path = os.path.join(os.path.dirname(__file__), path)
    with open(mapping_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw["label_mapping"]


def is_code_switched(lid_labels, min_tokens_per_lang=2):
    """
    Check whether a sample contains genuine Spanish-English code-switching.

    Criteria (matching the original preprocessing):
    - Must contain both lang1 (English) and lang2 (Spanish) tokens
    - Each language must have at least min_tokens_per_lang tokens
    """
    counts = Counter(lid_labels)
    return counts.get("lang1", 0) >= min_tokens_per_lang and \
           counts.get("lang2", 0) >= min_tokens_per_lang


def normalize_label(raw_label):
    """Map LINCE integer labels to sentiment strings."""
    # LINCE SA labels: 0=positive, 1=negative, 2=neutral
    if isinstance(raw_label, int):
        return ID2LABEL.get(raw_label)
    if isinstance(raw_label, str):
        label = raw_label.lower()
        if label in LABEL2ID:
            return label
    return None


def build_refined_dataset():
    """
    Main function to build the Refined Dataset.

    Steps:
    1. Load LINCE SA from Hugging Face
    2. Filter for genuine code-switching samples
    3. Assign sample IDs (sample_<global_index>)
    4. Apply label corrections from label_mapping.json
    5. Remove samples with duplicate IDs (unreliable annotations)
    6. Save to refined_dataset.json
    """
    print("Loading label corrections from label_mapping.json ...")
    label_mapping = load_label_mapping()
    print(f"  {len(label_mapping)} corrections loaded")

    print("\nLoading LINCE SA dataset from Hugging Face ...")
    dataset = load_dataset("lince", "sa_spaeng")

    # Enumerate samples across train + validation, assigning global indices.
    # The global index becomes the sample_id (sample_<idx>).
    # This matches the ID scheme used in the original preprocessing pipeline.
    all_samples = []
    global_idx = 0

    for split in SPLITS_TO_USE:
        split_data = dataset[split]
        print(f"  {split}: {len(split_data)} samples")

        for row in split_data:
            sample_id = f"sample_{global_idx}"
            global_idx += 1

            words = row["words"]
            lid = row["lid"]
            raw_label = row["sa"]

            # Skip samples that are not genuine code-switching
            if not is_code_switched(lid):
                continue

            sentiment = normalize_label(raw_label)
            if sentiment is None:
                continue

            # Apply human-verified correction if available
            original_sentiment = sentiment
            if sample_id in label_mapping:
                correction = label_mapping[sample_id]
                # Sanity check: original label should match
                if correction["original"] == sentiment:
                    sentiment = correction["corrected"]

            all_samples.append({
                "id": sample_id,
                "text": " ".join(words),
                "tokens": words,
                "lid_labels": lid,
                "sentiment": sentiment,
                "original_sentiment": original_sentiment,
                "label_corrected": sentiment != original_sentiment,
            })

    print(f"\nCode-switched samples collected: {len(all_samples)}")

    # Remove samples with duplicate IDs across splits.
    # Duplicates indicate the same sentence appearing in multiple splits,
    # making the label ambiguous — these are excluded from the Refined Dataset.
    id_counts = Counter(s["id"] for s in all_samples)
    duplicate_ids = {sid for sid, count in id_counts.items() if count > 1}
    print(f"Duplicate sample IDs removed: {len(duplicate_ids)}")

    refined = [s for s in all_samples if s["id"] not in duplicate_ids]
    print(f"Refined Dataset size: {len(refined)}")

    # Label distribution
    dist = Counter(s["sentiment"] for s in refined)
    print(f"\nLabel distribution:")
    for label, count in sorted(dist.items()):
        print(f"  {label}: {count} ({count/len(refined)*100:.1f}%)")

    corrections_applied = sum(1 for s in refined if s["label_corrected"])
    print(f"\nLabel corrections applied: {corrections_applied}")

    # Save output
    output_path = os.path.join(os.path.dirname(__file__), "refined_dataset.json")
    output = {
        "metadata": {
            "dataset": "LINCE SA Refined Dataset",
            "source": "load_dataset('lince', 'sa_spaeng') — Hugging Face",
            "paper": "A Relabeling Approach for Spanish-English Code-Switching SA (KSC 2025)",
            "total_samples": len(refined),
            "label_corrections_applied": corrections_applied,
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
