import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

df = pd.read_csv('dataset/meta_features.csv')
print('Total rows:', len(df))
print('Human score distribution:')
print(df['human_score'].value_counts().sort_index())
print()
print('raw_score_10 stats:')
print(df['raw_score_10'].describe())
print()
print('concept_score stats:')
print(df['concept_score'].describe())
print()
print('adj_mean stats:')
print(df['adj_mean'].describe())
print()

p, _ = pearsonr(df['human_score'], df['raw_score_10'])
s, _ = spearmanr(df['human_score'], df['raw_score_10'])
print(f'raw_score_10 vs human: pearson={p:.3f} spearman={s:.3f}')
p2, _ = pearsonr(df['human_score'], df['concept_score'])
s2, _ = spearmanr(df['human_score'], df['concept_score'])
print(f'concept_score vs human: pearson={p2:.3f} spearman={s2:.3f}')
p3, _ = pearsonr(df['human_score'], df['adj_mean'])
s3, _ = spearmanr(df['human_score'], df['adj_mean'])
print(f'adj_mean vs human: pearson={p3:.3f} spearman={s3:.3f}')
print()

print('zero_credit_ratio stats:')
print(df['zero_credit_ratio'].describe())
print()

bands = [(1,3),(4,6),(7,8),(9,10)]
for lo, hi in bands:
    mask = (df['human_score'] >= lo) & (df['human_score'] <= hi)
    sub = df[mask]
    rs_mean = sub['raw_score_10'].mean()
    rs_std = sub['raw_score_10'].std()
    cs_mean = sub['concept_score'].mean()
    print(f'Human {lo}-{hi}: n={len(sub)} | raw_score_10 mean={rs_mean:.2f} std={rs_std:.2f} | concept mean={cs_mean:.1f}')
