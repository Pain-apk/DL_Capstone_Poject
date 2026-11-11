"""Phase 0 audit: template rate, ingredient split bugs, gitignore check"""
import json, re, pathlib, pandas as pd
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent

# --- 0a: JSONL template rate ---
jsonl_path = ROOT / 'data/processed/pilot_unverified/train.jsonl'
bad_items = 0
total_ingredients = 0
for line in open(jsonl_path, 'r', encoding='utf-8'):
    obj = json.loads(line)
    for msg in obj.get('messages', []):
        if msg.get('role') == 'assistant':
            try:
                recipe = json.loads(msg['content'])
                for ing in recipe.get('ingredients', []):
                    total_ingredients += 1
                    item = ing.get('item', '').strip()
                    qty = ing.get('quantity', '').strip()
                    # Bug: item='leaves', quantity='8 curry' (multi-word split)
                    # Or item='seeds' with qty containing multiple words
                    if item in ('leaves', 'seeds', 'clove') and len(qty.split()) > 1:
                        bad_items += 1
            except Exception:
                pass

print(f'0a: Total ingredients in JSONL: {total_ingredients}')
print(f'0a: Badly split (item=leaves/seeds/clove, qty=multi-word): {bad_items}')
print(f'0a: Bug rate: {100*bad_items/max(total_ingredients,1):.2f}%')

# CSV-level template rate
df = pd.read_csv(ROOT / 'data/raw/maharashtra_drafts.csv')
all_steps = []
for s in df['steps'].dropna():
    all_steps.extend([x.strip() for x in str(s).split('|') if x.strip()])
print(f'\n0a: CSV steps total={len(all_steps)}, unique={len(set(all_steps))}')
print(f'    Template (duplicated) rate: {100*(1-len(set(all_steps))/len(all_steps)):.1f}%')

ct = Counter(all_steps)
print('Top 5 most repeated steps:')
for s, c in ct.most_common(5):
    print(f'  [{c}x] {s[:100]}')

# Non-veg cook-through sentence count
nonveg = df[df['meat_type'].notna() & (df['meat_type'] != 'none')]
nv_steps = []
for s in nonveg['steps'].dropna():
    nv_steps.extend([x.strip() for x in str(s).split('|') if x.strip()])
nv_ct = Counter(nv_steps)
cook_through = [s for s, c in nv_ct.most_common() if 'tender' in s.lower() or 'cooked' in s.lower() or 'safe internal' in s.lower() or 'simmer' in s.lower()]
print(f'\n0a: Non-veg cook-through sentences ({len(cook_through)} candidates):')
for s in cook_through[:3]:
    print(f'  [{nv_ct[s]}x] {s[:120]}')

# --- 0b: ingredient_with_quantity split bugs ---
bad_entries = []
for _, row in df.iterrows():
    iq = str(row.get('ingredients_with_quantity', ''))
    for entry in iq.split('|'):
        entry = entry.strip()
        # Pattern: number, word, then 'leaves' or 'seeds' at end = wrongly split
        # e.g. '8 curry leaves' should be: qty='8', item='curry leaves'
        m = re.match(r'^(\d+(?:/\d+)?(?:\.\d+)?)\s+(\w+)\s+(leaves|seeds|clove)$', entry, re.I)
        if m:
            bad_entries.append({'dish': row['dish_name'], 'raw': entry,
                                'wrong_qty': m.group(1)+' '+m.group(2),
                                'wrong_item': m.group(3),
                                'correct_qty': m.group(1),
                                'correct_item': m.group(2)+' '+m.group(3)})

print(f'\n0b: Ingredient split bugs in CSV: {len(bad_entries)} entries across {len(set(e["dish"] for e in bad_entries))} dishes')
print('Sample fixes:')
for e in bad_entries[:5]:
    print(f'  Dish={e["dish"]}: "{e["raw"]}" -> qty="{e["correct_qty"]}" item="{e["correct_item"]}"')

# --- 0c: .gitignore check ---
print('\n0c: .gitignore check:')
gi = (ROOT / '.gitignore').read_text(encoding='utf-8')
required = ['venv/', 'data/cache/', 'outputs/', '*.safetensors', '*.gguf', '.env']
for r in required:
    present = r in gi
    print(f'  {"OK" if present else "MISSING"}: {r}')

# Check if any of these were committed
import subprocess
result = subprocess.run(['git', 'ls-files', '--cached'], capture_output=True, text=True, cwd=ROOT)
committed = result.stdout.strip().split('\n')
flagged = [f for f in committed if any(x in f for x in ['venv/', '.safetensors', '.gguf', 'data/cache/', 'outputs/'])]
print(f'\nCommitted files that should be gitignored: {flagged if flagged else "None"}')

