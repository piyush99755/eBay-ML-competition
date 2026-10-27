"""
Compare aspect distributions and aggregate stats between original Quiz submission
and Phase 26B filtered candidate submissions.
"""

import csv
from collections import Counter
from pathlib import Path

BASE_TSV = Path("submissions/submission_quiz_exp10_baseline_corrected.tsv")
MIN060_TSV = Path("submissions/submission_quiz_exp26b_min060.tsv")
MIN080_TSV = Path("submissions/submission_quiz_exp26b_min080.tsv")


def load_aspect_counts(tsv_path: Path):
    counts = Counter()
    total = 0
    unique_recs = set()
    cat_counts = Counter()
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        for row in reader:
            rec_id, cat_id, asp, val = int(row[0]), int(row[1]), row[2], row[3]
            counts[asp] += 1
            cat_counts[cat_id] += 1
            unique_recs.add(rec_id)
            total += 1
    return counts, cat_counts, total, len(unique_recs)


def main():
    base_counts, base_cats, base_tot, base_recs = load_aspect_counts(BASE_TSV)
    m60_counts, m60_cats, m60_tot, m60_recs = load_aspect_counts(MIN060_TSV)
    m80_counts, m80_cats, m80_tot, m80_recs = load_aspect_counts(MIN080_TSV)

    print("=" * 80)
    print(f"{'Aspect Name':<35}{'Base':<10}{'Min 0.60':<12}{'Min 0.80':<12}{'Drop (0.80)':<12}")
    print("=" * 80)

    for asp, b_cnt in base_counts.most_common():
        cnt_60 = m60_counts.get(asp, 0)
        cnt_80 = m80_counts.get(asp, 0)
        drop_pct = ((b_cnt - cnt_80) / b_cnt) * 100.0 if b_cnt > 0 else 0.0
        print(f"{asp:<35}{b_cnt:<10}{cnt_60:<12}{cnt_80:<12}{drop_pct:<6.1f}%")

    print("-" * 80)
    print(f"{'Total Rows':<35}{base_tot:<10}{m60_tot:<12}{m80_tot:<12}{((base_tot - m80_tot)/base_tot)*100:.2f}%")
    print(f"{'Unique Titles with Rows':<35}{base_recs:<10}{m60_recs:<12}{m80_recs:<12}")
    print(f"{'Zero-Entity Titles':<35}{25000 - base_recs:<10}{25000 - m60_recs:<12}{25000 - m80_recs:<12}")
    print(f"{'Category 1 Rows':<35}{base_cats[1]:<10}{m60_cats[1]:<12}{m80_cats[1]:<12}")
    print(f"{'Category 2 Rows':<35}{base_cats[2]:<10}{m60_cats[2]:<12}{m80_cats[2]:<12}")
    print("=" * 80)


if __name__ == "__main__":
    main()
