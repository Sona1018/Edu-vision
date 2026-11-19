"""
diagnose_separation.py
Analyze which features best separate human score bands.
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

df = pd.read_csv('dataset/meta_features.csv')

feature_cols = [c for c in df.columns if c not in ('question_id','answer_id','human_score','hybrid_score_100')]

print("=== Per-feature correlation with human_score ===")
corrs = []
for c in feature_cols:
    try:
        p, _ = pearsonr(df['human_score'], df[c].fillna(0))
        s, _ = spearmanr(df['human_score'], df[c].fillna(0))
        corrs.append((c, p, s))
    except Exception:
        pass

corrs.sort(key=lambda x: abs(x[1]), reverse=True)
for name, p, s in corrs[:20]:
    print(f"  {name:35s}  pearson={p:+.3f}  spearman={s:+.3f}")

print()
print("=== Score band separation (key features) ===")
key = ['raw_score_10','concept_score','adj_mean','adj_min','adj_std',
       'zero_credit_ratio','coverage_06','raw_sim_mean','raw_sim_min',
       'combined_mean','combined_min','semantic_score','off_topic_gate']
bands = [(1,3),(4,6),(7,8),(9,10)]
for feat in key:
    if feat not in df.columns:
        continue
    vals = []
    for lo, hi in bands:
        mask = (df['human_score'] >= lo) & (df['human_score'] <= hi)
        vals.append(df.loc[mask, feat].mean())
    spread = max(vals) - min(vals)
    print(f"  {feat:30s}  {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {vals[3]:.3f}  spread={spread:.3f}")
