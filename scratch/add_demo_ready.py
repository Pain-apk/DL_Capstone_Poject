import pandas as pd
df = pd.read_csv('data/raw/maharashtra_drafts.csv')
if 'demo_ready' not in df.columns:
    df['demo_ready'] = 'no'
    df.to_csv('data/raw/maharashtra_drafts.csv', index=False)
    print(f"Added demo_ready column (default 'no') to all {len(df)} rows.")
else:
    print(f"demo_ready column already exists. Value counts: {df['demo_ready'].value_counts().to_dict()}")

