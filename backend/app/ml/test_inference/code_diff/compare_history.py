"""
compare_history.py - diff two history.jsonl files (CLI run vs backend session run) frame by frame.

    python compare_history.py cli_history.jsonl session_history.jsonl [--width-tol 0.5] [--conf-tol 0.02]

Both files are the wheel's own per-frame result dict, so every difference printed here is a
real difference in what the wheel returned (``fps`` is wall-clock timing and is ignored).
Frames are matched by their ``frame`` number.
"""
import argparse
import json
import sys


def load(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["frame"]] = r
    return out


def tails(r):
    return {d["id"]: d for d in r.get("detections", []) if d["role"] == "TAIL"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="history.jsonl of run A (e.g. CLI)")
    ap.add_argument("b", help="history.jsonl of run B (e.g. session)")
    ap.add_argument("--width-tol", type=float, default=0.5, help="px; width_px differences above this are reported")
    ap.add_argument("--conf-tol", type=float, default=0.02)
    a = ap.parse_args()

    A, B = load(a.a), load(a.b)
    common = sorted(set(A) & set(B))
    print(f"A: {len(A)} frames   B: {len(B)} frames   in both: {len(common)}")
    if set(A) ^ set(B):
        only = sorted(set(A) ^ set(B))
        print(f"frames present in only one run: {len(only)} (first: {only[:10]})")

    count_diff, verdict_flips, bigger_diff, width_diff, conf_diff, id_diff = [], [], [], [], [], []
    for n in common:
        x, y = A[n], B[n]
        if (x["heads_count"], x["tails_count"]) != (y["heads_count"], y["tails_count"]):
            count_diff.append((n, (x["heads_count"], x["tails_count"]), (y["heads_count"], y["tails_count"])))
        tx, ty = tails(x), tails(y)
        if set(tx) != set(ty):
            id_diff.append(n)
        for i in set(tx) & set(ty):
            if tx[i]["size"] != ty[i]["size"]:
                verdict_flips.append((n, i, tx[i]["size"], ty[i]["size"], tx[i]["width_px"], ty[i]["width_px"]))
            if tx[i]["width_px"] and ty[i]["width_px"] and abs(tx[i]["width_px"] - ty[i]["width_px"]) > a.width_tol:
                width_diff.append((n, i, tx[i]["width_px"], ty[i]["width_px"]))
            if abs(tx[i]["conf"] - ty[i]["conf"]) > a.conf_tol:
                conf_diff.append((n, i, tx[i]["conf"], ty[i]["conf"]))
        bx, by = (x.get("bigger_tube") or {}), (y.get("bigger_tube") or {})
        if (bx.get("status"), bx.get("id")) != (by.get("status"), by.get("id")):
            bigger_diff.append((n, bx.get("status"), bx.get("id"), by.get("status"), by.get("id")))

    def show(title, rows, k=8):
        print(f"\n{title}: {len(rows)}")
        for r in rows[:k]:
            print("   ", r)
        if len(rows) > k:
            print(f"    ... +{len(rows) - k} more")

    show("frames with different head/tail COUNT  (frame, A(h,t), B(h,t))", count_diff)
    show("frames where the set of tail track IDs differs", id_diff)
    show("tail Bigger/Smaller/Same FLIPS  (frame, id, A, B, A width, B width)", verdict_flips)
    show("bigger_tube differs  (frame, A status, A id, B status, B id)", bigger_diff)
    show(f"tail width_px differs > {a.width_tol}px  (frame, id, A, B)", width_diff)
    show(f"tail conf differs > {a.conf_tol}  (frame, id, A, B)", conf_diff)

    identical = not (count_diff or id_diff or verdict_flips or bigger_diff)
    print("\nRESULT:", "runs are identical in counts, ids, verdicts and bigger_tube" if identical else
          "runs differ -> the wheel returned different results for the same frame; compare model numerics / decode / settings, not the backend glue")
    sys.exit(0 if identical else 1)


if __name__ == "__main__":
    main()
