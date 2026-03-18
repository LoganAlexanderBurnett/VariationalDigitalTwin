import os

import pandas as pd


def shrink_train_set(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    n_removed: int,
    seq_len: int = 5485,
):
    """Remove the last ``n_removed * seq_len`` rows from the training split."""
    print(f"\n--- N = {n_removed} ---")
    print("Before lengths:")
    print(f"  train: {len(train)}")
    print(f"  valid: {len(valid)}")
    print(f"  test : {len(test)}")

    step = n_removed * seq_len
    train_new = train.iloc[:-step].reset_index(drop=True)

    print("After lengths:")
    print(f"  train: {len(train_new)}")
    print(f"  valid: {len(valid)} (unchanged)")
    print(f"  test : {len(test)}  (unchanged)")

    output_dir = f"shrink_train_{n_removed}"
    os.makedirs(output_dir, exist_ok=True)
    train_new.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    valid.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print(f"Saved splits to '{output_dir}/'")

    return train_new, valid, test


def main():
    train_orig = pd.read_csv('TF_TS_train.csv')
    valid_orig = pd.read_csv('TF_TS_valid.csv')
    test_orig = pd.read_csv('TF_TS_test.csv')

    for df in (train_orig, valid_orig, test_orig):
        df['TS'] = df['TS'].interpolate()

    for n_removed in range(1, 63):
        shrink_train_set(train_orig, valid_orig, test_orig, n_removed)


if __name__ == '__main__':
    main()
