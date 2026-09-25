"""
make_gray_dataset.py - copy a YOLO dataset, converting every image to 3-channel grayscale.

Train on grayscale so the model learns shape (connector geometry, cut end, walls)
instead of tube tint or lighting colour. Labels are copied unchanged.

  python make_gray_dataset.py --src datasets/tube_ends --dst datasets/tube_ends_gray
"""
import argparse
import shutil
from pathlib import Path

import cv2

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    n = 0
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() in IMG_EXT:
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if g is None:                     # unreadable or not really an image
                print(f"[skip] cannot read {p}")
                continue
            cv2.imwrite(str(out), cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
            n += 1
        else:
            shutil.copy2(p, out)          # labels, yaml, etc.
    print(f"{n} images -> {dst}")


if __name__ == "__main__":
    main()