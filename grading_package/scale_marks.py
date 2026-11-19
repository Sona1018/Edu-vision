#!/usr/bin/env python3
"""
scale_marks.py — Dynamic Marks Scaler for IntelliPlace Grading Dataset
----------------------------------------------------------------------
Usage:
    python scale_marks.py --scale 100   # Full 100-mark paper
    python scale_marks.py --scale 50    # Half (50-mark) paper
    python scale_marks.py --scale 75    # Custom scale

This reads grading_benchmark_dataset.csv, recalculates all marks
based on the target scale, and saves a new file.
"""
import argparse
import csv
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Scale marks in grading dataset")
    parser.add_argument("--scale", type=float, default=100, help="Target total paper marks (default: 100)")
    parser.add_argument("--input", default="grading_benchmark_dataset.csv", help="Input CSV file")
    parser.add_argument("--output", default=None, help="Output CSV file (default: auto-named)")
    parser.add_argument("--rubric", default="master_reference_rubric.json", help="Rubric JSON file")
    args = parser.parse_args()
    
    scale = args.scale
    if args.output is None:
        args.output = f"grading_dataset_scale_{int(scale)}.csv"
    
    # Load rubric for weight info
    with open(args.rubric, 'r', encoding='utf-8') as f:
        rubric = json.load(f)
    
    weight_map = {q["question_id"]: q["weight_pct"] for q in rubric["questions"]}
    
    # Read and rescale
    with open(args.input, 'r', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)
    
    for row in rows:
        qid = row["Question_ID"]
        weight = float(weight_map.get(qid, row.get("Weight_Pct", 20)))
        max_marks = weight * scale / 100
        est_pct = float(row.get("Estimated_Score_Pct", 0))
        
        row["Max_Marks_Scaled"] = round(max_marks, 2)
        row["Expected_Score_Scaled"] = round(est_pct / 100 * max_marks, 2)
        row["Paper_Scale"] = int(scale)
    
    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.output, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✅ Scaled to {int(scale)}-mark paper")
    print(f"   Input:  {args.input}")
    print(f"   Output: {args.output}")
    print(f"   Rows:   {len(rows)}")

if __name__ == "__main__":
    main()
