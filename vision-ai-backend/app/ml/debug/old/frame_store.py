# """
# frame_store.py -- standalone SQLite per-frame logger for DuckAnalyzer.

# Bolt-on, NOT a change to duck_analyzer.py. It consumes the result dict your
# analyzer already returns from process_frame() and writes one row per frame plus
# one row per detection (with an optional embedding BLOB) to a SQLite file. Your
# analyzer logic is untouched -- you just call store.log(result, embeddings) once
# after each process_frame.

#     from frame_store import FrameStore
#     store = FrameStore("duck_run.db", run_id="video1")     # opens/creates db
#     ...
#     result = analyzer.process_frame(frame)
#     store.log(result, embeddings={1: vec1, 2: vec2, ...})  # embeddings optional
#     ...
#     store.close()

# `embeddings` maps a detection key -> np.float32 vector (from
# DeepEmbedder.embed). Ducks and other_toys have SEPARATE id spaces (a duck can
# be #1 and a toy can also be #1), so the key must be the (species, id) tuple,
# e.g. {("duck", 1): vec, ("other_toys", 1): vec2}. A plain int id is also
# accepted and treated as ("duck", id) for convenience. Any detection not in the
# map is stored with a NULL embedding. Everything is best-effort: a logging error
# is caught and printed, never raised into your pipeline.

# Schema
# ------
# runs(run_id TEXT, expected INT, started REAL)
# frames(run_id, frame INT, status TEXT, fps REAL, detected INT, expected INT,
#        missing_count INT, added_count INT, other_count INT, reasons TEXT,
#        anchor_locked INT, hand INT, ts REAL)
# detections(run_id, frame INT, display_id INT, species TEXT, status TEXT,
#            x1 INT, y1 INT, x2 INT, y2 INT, confidence REAL,
#            excess INT, embedding BLOB)
# """

# import time
# import json
# import sqlite3
# import threading

# import numpy as np


# def _emb_to_blob(v):
#     if v is None:
#         return None
#     return np.asarray(v, dtype=np.float32).tobytes()


# class FrameStore:
#     def __init__(self, db_path="duck_run.db", run_id=None, expected=None,
#                  store_embeddings=True):
#         self.db_path = db_path
#         self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
#         self.store_embeddings = bool(store_embeddings)
#         self._lock = threading.Lock()
#         # check_same_thread=False so a Celery/threaded backend can share it;
#         # we serialize writes with our own lock.
#         self.conn = sqlite3.connect(db_path, check_same_thread=False)
#         self.conn.execute("PRAGMA journal_mode=WAL;")
#         self._init_schema()
#         self._register_run(expected)

#     def _init_schema(self):
#         c = self.conn
#         c.execute("""CREATE TABLE IF NOT EXISTS runs(
#             run_id TEXT PRIMARY KEY, expected INTEGER, started REAL)""")
#         c.execute("""CREATE TABLE IF NOT EXISTS frames(
#             run_id TEXT, frame INTEGER, status TEXT, fps REAL,
#             detected INTEGER, expected INTEGER, missing_count INTEGER,
#             added_count INTEGER, other_count INTEGER, reasons TEXT,
#             anchor_locked INTEGER, hand INTEGER, ts REAL,
#             PRIMARY KEY(run_id, frame))""")
#         c.execute("""CREATE TABLE IF NOT EXISTS detections(
#             run_id TEXT, frame INTEGER, display_id INTEGER, species TEXT,
#             status TEXT, x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
#             confidence REAL, excess INTEGER, embedding BLOB)""")
#         c.execute("""CREATE INDEX IF NOT EXISTS idx_det_run_frame
#                      ON detections(run_id, frame)""")
#         c.execute("""CREATE INDEX IF NOT EXISTS idx_det_run_id
#                      ON detections(run_id, display_id)""")
#         c.commit()

#     def _register_run(self, expected):
#         try:
#             with self._lock:
#                 self.conn.execute(
#                     "INSERT OR IGNORE INTO runs(run_id, expected, started) "
#                     "VALUES(?,?,?)", (self.run_id, expected, time.time()))
#                 self.conn.commit()
#         except Exception as e:
#             print(f"[DB] [WARN] register run failed: {e}")

#     def log(self, result, embeddings=None):
#         """Write one frame row + its detection rows.
#         result:     the dict returned by DuckAnalyzer.process_frame().
#         embeddings: optional {display_id: np.float32 vector} for this frame.
#         Best-effort; never raises."""
#         if result is None:
#             return
#         embeddings = embeddings or {}
#         try:
#             frame = int(result.get("frame", 0))
#             with self._lock:
#                 self.conn.execute(
#                     "INSERT OR REPLACE INTO frames(run_id, frame, status, fps, "
#                     "detected, expected, missing_count, added_count, "
#                     "other_count, reasons, anchor_locked, hand, ts) "
#                     "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
#                     (self.run_id, frame,
#                      result.get("status"),
#                      float(result.get("fps", 0.0) or 0.0),
#                      int(result.get("detected_duck_count", 0) or 0),
#                      result.get("expected_duck_count"),
#                      int(result.get("missing_count", 0) or 0),
#                      int(result.get("added_count", 0) or 0),
#                      int(result.get("other_count", 0) or 0),
#                      json.dumps(result.get("reasons", [])),
#                      1 if result.get("anchor_locked") else 0,
#                      1 if result.get("hand_detected") else 0,
#                      time.time()))

#                 rows = []
#                 for det in result.get("detections", []):
#                     bbox = det.get("bbox", [0, 0, 0, 0])
#                     x1, y1, x2, y2 = (list(bbox) + [0, 0, 0, 0])[:4]
#                     did = det.get("id", -1)
#                     species = det.get("species")
#                     emb = None
#                     if self.store_embeddings and embeddings:
#                         # prefer the (species, id) key; fall back to a bare int
#                         # id (treated as a duck) for callers that pass ints.
#                         emb = embeddings.get((species, did))
#                         if emb is None:
#                             emb = embeddings.get(did)
#                     rows.append((
#                         self.run_id, frame,
#                         int(did) if did is not None else -1,
#                         det.get("species"),
#                         det.get("status"),
#                         int(x1), int(y1), int(x2), int(y2),
#                         float(det.get("confidence", 0.0) or 0.0),
#                         1 if det.get("excess") else 0,
#                         _emb_to_blob(emb)))
#                 if rows:
#                     self.conn.executemany(
#                         "INSERT INTO detections(run_id, frame, display_id, "
#                         "species, status, x1, y1, x2, y2, confidence, excess, "
#                         "embedding) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", rows)
#                 self.conn.commit()
#         except Exception as e:
#             print(f"[DB] [WARN] frame log failed (non-fatal): {e}")

#     # ---------------- read helpers (for later analysis) ---------------- #
#     def get_embeddings_for_id(self, display_id, species="duck"):
#         """Return [(frame, np.float32 vector), ...] for one (species, id)."""
#         cur = self.conn.execute(
#             "SELECT frame, embedding FROM detections WHERE run_id=? AND "
#             "species=? AND display_id=? AND embedding IS NOT NULL ORDER BY frame",
#             (self.run_id, species, int(display_id)))
#         out = []
#         for frame, blob in cur.fetchall():
#             out.append((frame, np.frombuffer(blob, dtype=np.float32)))
#         return out

#     def close(self):
#         try:
#             with self._lock:
#                 self.conn.commit()
#                 self.conn.close()
#         except Exception:
#             pass


"""
frame_store.py -- standalone SQLite per-frame logger for DuckAnalyzer.

Bolt-on, NOT a change to duck_analyzer.py. It consumes the result dict your
analyzer already returns from process_frame() and writes one row per frame plus
one row per detection (with an optional embedding BLOB) to a SQLite file. Your
analyzer logic is untouched -- you just call store.log(result, embeddings) once
after each process_frame.

    from frame_store import FrameStore
    store = FrameStore("duck_run.db", run_id="video1")     # opens/creates db
    ...
    result = analyzer.process_frame(frame)
    store.log(result, embeddings={1: vec1, 2: vec2, ...})  # embeddings optional
    ...
    store.close()

`embeddings` maps a detection key -> np.float32 vector (from
DeepEmbedder.embed). Ducks and other_toys have SEPARATE id spaces (a duck can
be #1 and a toy can also be #1), so the key must be the (species, id) tuple,
e.g. {("duck", 1): vec, ("other_toys", 1): vec2}. A plain int id is also
accepted and treated as ("duck", id) for convenience. Any detection not in the
map is stored with a NULL embedding. Everything is best-effort: a logging error
is caught and printed, never raised into your pipeline.

Schema
------
runs(run_id TEXT, expected INT, started REAL)
frames(run_id, frame INT, status TEXT, fps REAL, detected INT, expected INT,
       missing_count INT, added_count INT, other_count INT, reasons TEXT,
       anchor_locked INT, hand INT, ts REAL)
detections(run_id, frame INT, display_id INT, species TEXT, status TEXT,
           x1 INT, y1 INT, x2 INT, y2 INT, confidence REAL,
           excess INT, embedding BLOB)
"""

import time
import json
import sqlite3
import threading

import numpy as np


def _emb_to_blob(v):
    if v is None:
        return None
    return np.asarray(v, dtype=np.float32).tobytes()


class FrameStore:
    def __init__(self, db_path="duck_run.db", run_id=None, expected=None,
                 store_embeddings=True):
        self.db_path = db_path
        self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
        self.store_embeddings = bool(store_embeddings)
        self._lock = threading.Lock()
        # check_same_thread=False so a Celery/threaded backend can share it;
        # we serialize writes with our own lock.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()
        self._register_run(expected)

    def _init_schema(self):
        c = self.conn
        c.execute("""CREATE TABLE IF NOT EXISTS runs(
            run_id TEXT PRIMARY KEY, expected INTEGER, started REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS frames(
            run_id TEXT, frame INTEGER, status TEXT, fps REAL,
            detected INTEGER, expected INTEGER, missing_count INTEGER,
            added_count INTEGER, other_count INTEGER, reasons TEXT,
            anchor_locked INTEGER, hand INTEGER, ts REAL,
            PRIMARY KEY(run_id, frame))""")
        c.execute("""CREATE TABLE IF NOT EXISTS detections(
            run_id TEXT, frame INTEGER, display_id INTEGER, species TEXT,
            status TEXT, x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
            confidence REAL, excess INTEGER, embedding BLOB)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_det_run_frame
                     ON detections(run_id, frame)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_det_run_id
                     ON detections(run_id, display_id)""")
        c.commit()

    def _register_run(self, expected):
        try:
            with self._lock:
                self.conn.execute(
                    "INSERT OR IGNORE INTO runs(run_id, expected, started) "
                    "VALUES(?,?,?)", (self.run_id, expected, time.time()))
                self.conn.commit()
        except Exception as e:
            print(f"[DB] [WARN] register run failed: {e}")

    def log(self, result, embeddings=None):
        """Write one frame row + its detection rows.
        result:     the dict returned by DuckAnalyzer.process_frame().
        embeddings: optional {display_id: np.float32 vector} for this frame.
        Best-effort; never raises."""
        if result is None:
            return
        embeddings = embeddings or {}
        try:
            frame = int(result.get("frame", 0))
            with self._lock:
                self.conn.execute(
                    "INSERT OR REPLACE INTO frames(run_id, frame, status, fps, "
                    "detected, expected, missing_count, added_count, "
                    "other_count, reasons, anchor_locked, hand, ts) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (self.run_id, frame,
                     result.get("status"),
                     float(result.get("fps", 0.0) or 0.0),
                     int(result.get("detected_duck_count", 0) or 0),
                     result.get("expected_duck_count"),
                     int(result.get("missing_count", 0) or 0),
                     int(result.get("added_count", 0) or 0),
                     int(result.get("other_count", 0) or 0),
                     json.dumps(result.get("reasons", [])),
                     1 if result.get("anchor_locked") else 0,
                     1 if result.get("hand_detected") else 0,
                     time.time()))

                rows = []
                for det in result.get("detections", []):
                    bbox = det.get("bbox", [0, 0, 0, 0])
                    x1, y1, x2, y2 = (list(bbox) + [0, 0, 0, 0])[:4]
                    did = det.get("id", -1)
                    species = det.get("species")
                    emb = None
                    if self.store_embeddings and embeddings:
                        # prefer the (species, id) key; fall back to a bare int
                        # id (treated as a duck) for callers that pass ints.
                        emb = embeddings.get((species, did))
                        if emb is None:
                            emb = embeddings.get(did)
                    rows.append((
                        self.run_id, frame,
                        int(did) if did is not None else -1,
                        det.get("species"),
                        det.get("status"),
                        int(x1), int(y1), int(x2), int(y2),
                        float(det.get("confidence", 0.0) or 0.0),
                        1 if det.get("excess") else 0,
                        _emb_to_blob(emb)))
                if rows:
                    self.conn.executemany(
                        "INSERT INTO detections(run_id, frame, display_id, "
                        "species, status, x1, y1, x2, y2, confidence, excess, "
                        "embedding) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                self.conn.commit()
        except Exception as e:
            print(f"[DB] [WARN] frame log failed (non-fatal): {e}")

    # ---------------- read helpers (for later analysis) ---------------- #
    def get_embeddings_for_id(self, display_id, species="duck"):
        """Return [(frame, np.float32 vector), ...] for one (species, id)."""
        cur = self.conn.execute(
            "SELECT frame, embedding FROM detections WHERE run_id=? AND "
            "species=? AND display_id=? AND embedding IS NOT NULL ORDER BY frame",
            (self.run_id, species, int(display_id)))
        out = []
        for frame, blob in cur.fetchall():
            out.append((frame, np.frombuffer(blob, dtype=np.float32)))
        return out

    def close(self):
        try:
            with self._lock:
                self.conn.commit()
                self.conn.close()
        except Exception:
            pass