"""Minimal CSV run logger — append-only, one row per logged step/epoch."""
import csv
import os


class CSVLogger:
    def __init__(self, log_dir, name="metrics"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{name}.csv")
        self._header = None
        self._f = open(self.path, "w", newline="")
        self._w = csv.writer(self._f)

    def log(self, row: dict):
        if self._header is None:
            self._header = list(row.keys())
            self._w.writerow(self._header)
        self._w.writerow([row.get(k, "") for k in self._header])
        self._f.flush()

    def close(self):
        self._f.close()
