import os
import subprocess
import re

def find_shrink_dirs(base="."):
    """
    Return a list of all subdirectories named 'shrink_train_<N>',
    sorted by the integer N.
    """
    pattern = re.compile(r"shrink_train_(\d+)$")
    dirs = []
    for d in os.listdir(base):
        m = pattern.match(d)
        if m and os.path.isdir(os.path.join(base, d)):
            num = int(m.group(1))
            dirs.append((num, d))
    # sort by the extracted number, then return just the names
    return [d for _, d in sorted(dirs, key=lambda x: x[0])]

def main():
    dirs = find_shrink_dirs()
    total = len(dirs)
    if total == 0:
        print("No 'shrink_train_*' directories found.")
        return

    for idx, d in enumerate(dirs, start=1):
        print(f"[{idx}/{total}] Processing '{d}'…")
        cmd = ["python", "run_metrics.py", "--data-dir", d]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running metrics in {d}: {e}")
            break

    print("\nAll done!")

if __name__ == "__main__":
    main()
