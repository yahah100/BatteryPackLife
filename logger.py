from __future__ import annotations

import csv
import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


def _sanitize_run_name(name: str | None) -> str:
    if not name:
        return "run"
    safe = name.strip().replace(os.sep, "_") if os.sep in name else name.strip()
    return safe.replace(" ", "_") or "run"


def _convert_value(value: Any) -> Any:
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except (TypeError, ValueError):  # pragma: no cover - defensive fallback
            pass
    if isinstance(value, (list, tuple)):
        return [_convert_value(v) for v in value]
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return value.tolist()
        except (TypeError, ValueError):  # pragma: no cover - defensive fallback
            pass
    return value


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)

    text = str(value)
    if not text:
        return '""'

    special_chars = ":#{}[],&*?|-<>=!%@\"'\t\n\r"
    if any(ch in text for ch in special_chars) or text.strip() != text:
        return json.dumps(text, ensure_ascii=False)
    return text


def _serialize_yaml(data: Any, indent: int = 0) -> Iterable[str]:
    prefix = " " * indent
    if isinstance(data, Mapping):
        for key in sorted(data):
            value = data[key]
            key_text = str(key)
            if isinstance(value, (Mapping, list, tuple)):
                yield f"{prefix}{key_text}:"
                yield from _serialize_yaml(value, indent + 2)
            else:
                yield f"{prefix}{key_text}: {_format_scalar(value)}"
    elif isinstance(data, (list, tuple)):
        for item in data:
            if isinstance(item, (Mapping, list, tuple)):
                yield f"{prefix}-"
                yield from _serialize_yaml(item, indent + 2)
            else:
                yield f"{prefix}- {_format_scalar(item)}"
    else:
        yield f"{prefix}{_format_scalar(data)}"


class Logger:
    """Persist training metadata with minimal dependencies.

    Metrics are appended to ``metrics.csv`` while hyperparameters are stored in
    ``hparams.yml`` for later inspection or experiment tracking.
    """

    def __init__(self, root_dir: str | Path, run_name: str | None = None) -> None:
        self._root_dir = Path(root_dir).expanduser().resolve()
        self._root_dir.mkdir(parents=True, exist_ok=True)

        safe_name = _sanitize_run_name(run_name)
        timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        tentative_dir = self._root_dir / f"{safe_name}-{timestamp}"
        counter = 1
        while tentative_dir.exists():
            tentative_dir = self._root_dir / f"{safe_name}-{timestamp}-{counter}"
            counter += 1

        self.run_dir = tentative_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.run_dir / "metrics.csv"
        self.hparams_path = self.run_dir / "hparams.yml"
        self._fieldnames: list[str] | None = None

    def log_hparams(self, params: Mapping[str, Any] | Any) -> None:
        if params is None:
            return

        if isinstance(params, Mapping):
            payload = {str(k): _convert_value(v) for k, v in params.items()}
        elif hasattr(params, "__dict__"):
            payload = {str(k): _convert_value(v) for k, v in vars(params).items()}
        else:
            raise TypeError("Hyperparameters must be a mapping or an object with __dict__")

        payload["logged_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        with self.hparams_path.open("w", encoding="utf-8") as stream:
            stream.write("---\n")
            for line in _serialize_yaml(payload):
                stream.write(f"{line}\n")

    def log(self, metrics: Mapping[str, Any]) -> None:
        if not metrics:
            return

        row = {str(k): _convert_value(v) for k, v in metrics.items()}

        rewrite_rows: list[dict[str, Any]] = []
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            writer_mode = "w"
        else:
            new_keys = [key for key in row if key not in self._fieldnames]
            if new_keys:
                self._fieldnames.extend(new_keys)
                writer_mode = "w"
                if self.metrics_path.exists():
                    with self.metrics_path.open("r", newline="", encoding="utf-8") as handle:
                        reader = csv.DictReader(handle)
                        rewrite_rows = list(reader)
            else:
                writer_mode = "a"

        for key in self._fieldnames:
            row.setdefault(key, "")

        if writer_mode == "w":
            with self.metrics_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
                writer.writeheader()
                for existing_row in rewrite_rows:
                    for key in self._fieldnames:
                        existing_row.setdefault(key, "")
                    writer.writerow(existing_row)
                writer.writerow(row)
        else:
            with self.metrics_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
                writer.writerow(row)

    def finish(self) -> None:
        if not self.metrics_path.exists():
            with self.metrics_path.open("w", newline="", encoding="utf-8") as handle:
                handle.write("epoch\n")
        if not self.hparams_path.exists():
            self.log_hparams({})
