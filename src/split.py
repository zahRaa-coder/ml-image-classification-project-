"""Step 3: stratified 70/15/15 split -> data/splits.csv (skips rejected images).

Leakage protection:
  * exact / near-identical copies were already removed globally by clean.py (pHash);
  * near_duplicates.py groups visually near-identical images (CNN cosine >= 0.92);
    each group is kept inside ONE split (StratifiedGroupKFold), so no near-copy can
    appear in both train and test;
  * a final check counts near-duplicate pairs that cross splits and fails if > 0.
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from config import CLEAN_DIR, SLUGS, SPLIT_CSV, REJECTED_TXT, SEED, ROOT, DATA, RESULTS


def load_rejected():
    if REJECTED_TXT.exists():
        return {l.strip() for l in REJECTED_TXT.read_text().splitlines() if l.strip()}
    return set()


def main():
    rej = load_rejected()
    rows = [(str(p.relative_to(ROOT)), s) for s in SLUGS
            for p in sorted((CLEAN_DIR / s).glob("*.jpg"))
            if str(p.relative_to(ROOT)) not in rej]
    df = pd.DataFrame(rows, columns=["path", "label"])
    gfile = DATA / "dup_groups.csv"
    if gfile.exists():
        g = pd.read_csv(gfile).set_index("path")["group"]
        df["group"] = df.path.map(g)
        missing = df.group.isna()
        df.loc[missing, "group"] = -(df.index[missing] + 1)    # unseen image -> own group
    else:
        print("WARNING: run near_duplicates.py first for group-aware splitting")
        df["group"] = range(len(df))

    # 20 folds -> 14 train / 3 val / 3 test  (= 70 / 15 / 15), stratified by class
    folds = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    df["fold"] = -1
    for k, (_, idx) in enumerate(folds.split(df, df.label, df.group)):
        df.loc[idx, "fold"] = k
    df["split"] = df.fold.map(lambda k: "test" if k < 3 else "val" if k < 6 else "train")
    df.drop(columns="fold").to_csv(SPLIT_CSV, index=False)

    # ---- leakage check
    leaks = 0
    pfile = DATA / "near_duplicates.csv"
    if pfile.exists():
        sp = df.set_index("path").split
        pairs = pd.read_csv(pfile)
        pairs = pairs[pairs.a.isin(sp.index) & pairs.b.isin(sp.index)]
        leaks = int((pairs.a.map(sp) != pairs.b.map(sp)).sum())
    table = df.groupby(["label", "split"]).size().unstack()[["train", "val", "test"]]
    print(f"removed by review: {len(rej)}")
    print(table); print(df.split.value_counts(normalize=True).round(3))
    print(f"near-duplicate pairs crossing splits: {leaks}")
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "split_counts.csv")
    assert leaks == 0, "data leakage detected between splits"


if __name__ == "__main__":
    main()
