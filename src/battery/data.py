from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


DEFAULT_DATE_FORMAT = "%d-%b-%Y %H:%M:%S"


@dataclass(frozen=True)
class BatteryRun:
    file_name: str
    path: Path
    date: str
    timestamp: datetime
    current: np.ndarray
    voltage: np.ndarray
    time: np.ndarray | None = None

    def cropped(self, length: int) -> "BatteryRun":
        return BatteryRun(
            file_name=self.file_name,
            path=self.path,
            date=self.date,
            timestamp=self.timestamp,
            current=self.current[:length].astype(np.float32, copy=False),
            voltage=self.voltage[:length].astype(np.float32, copy=False),
            time=None if self.time is None else self.time[:length],
        )

    def padded_or_truncated(self, length: int) -> "BatteryRun":
        current = self.current.astype(np.float32, copy=False)
        voltage = self.voltage.astype(np.float32, copy=False)
        if current.size < length:
            pad = length - current.size
            current = np.pad(current, (0, pad), mode="edge")
            voltage = np.pad(voltage, (0, pad), mode="edge")
        return BatteryRun(
            file_name=self.file_name,
            path=self.path,
            date=self.date,
            timestamp=self.timestamp,
            current=current[:length],
            voltage=voltage[:length],
            time=None if self.time is None else self.time[:length],
        )


@dataclass(frozen=True)
class StaticTrainTestSplit:
    train_runs: tuple[BatteryRun, ...]
    test_runs: tuple[BatteryRun, ...]

    def load_train_arrays(self, length: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
        currents: list[np.ndarray] = []
        voltages: list[np.ndarray] = []
        dates: list[str] = []
        for run in self.train_runs:
            cropped = run.cropped(length)
            if cropped.current.size == length and cropped.voltage.size == length:
                currents.append(cropped.current)
                voltages.append(cropped.voltage)
                dates.append(cropped.date)
        return np.stack(currents), np.stack(voltages), dates

    def iter_test_runs(self) -> Iterator[BatteryRun]:
        yield from self.test_runs


@dataclass(frozen=True)
class RollingBlock:
    runs: tuple[BatteryRun, ...]

    def load_arrays(self, seq_len: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
        padded_runs = [run.padded_or_truncated(seq_len) for run in self.runs]
        currents = np.stack([run.current for run in padded_runs])
        voltages = np.stack([run.voltage for run in padded_runs])
        dates = [run.date for run in padded_runs]
        return currents, voltages, dates


@dataclass(frozen=True)
class BatteryRunDataset:
    root_dir: Path
    runs: tuple[BatteryRun, ...]
    removed_runs: tuple[str, ...] = ()

    @classmethod
    def from_directory(
        cls,
        npz_dir: str | Path,
        date_fmt: str = DEFAULT_DATE_FORMAT,
        filter_negative_current: bool = True,
    ) -> "BatteryRunDataset":
        root_dir = Path(npz_dir)
        discovered_files = discover_run_files(root_dir)
        removed: list[str] = []
        admitted_runs: list[BatteryRun] = []

        for path in discovered_files:
            run = load_battery_run(path, date_fmt=date_fmt)
            if filter_negative_current and has_negative_current(run):
                removed.append(run.file_name)
                continue
            admitted_runs.append(run)

        sorted_runs = tuple(sort_runs_chronologically(admitted_runs))
        return cls(root_dir=root_dir, runs=sorted_runs, removed_runs=tuple(removed))

    def static_split(self, train_count: int) -> StaticTrainTestSplit:
        return StaticTrainTestSplit(
            train_runs=self.runs[:train_count],
            test_runs=self.runs[train_count:],
        )

    def rolling_block(self, file_names: Sequence[str]) -> RollingBlock:
        run_map = {run.file_name: run for run in self.runs}
        return RollingBlock(runs=tuple(run_map[file_name] for file_name in file_names))

    @property
    def files(self) -> list[str]:
        return [run.file_name for run in self.runs]


def discover_run_files(npz_dir: str | Path) -> list[Path]:
    root = Path(npz_dir)
    return sorted(path for path in root.iterdir() if path.name.startswith("run_") and path.suffix == ".npz")


def parse_run_date(date_str: str, date_fmt: str = DEFAULT_DATE_FORMAT) -> datetime:
    return datetime.strptime(date_str, date_fmt)


def load_battery_run(path: str | Path, date_fmt: str = DEFAULT_DATE_FORMAT) -> BatteryRun:
    path = Path(path)
    data = np.load(path)
    date = str(data["date"].item())
    current = np.asarray(data["current"], dtype=np.float32)
    voltage = np.asarray(data["voltage"], dtype=np.float32)
    time = None
    if "time" in data.files:
        time = np.asarray(data["time"])
    return BatteryRun(
        file_name=path.name,
        path=path,
        date=date,
        timestamp=parse_run_date(date, date_fmt=date_fmt),
        current=current,
        voltage=voltage,
        time=time,
    )


def has_negative_current(run: BatteryRun) -> bool:
    return bool(np.any(run.current < 0))


def filter_negative_current_runs(runs: Iterable[BatteryRun]) -> tuple[list[BatteryRun], list[BatteryRun]]:
    admitted: list[BatteryRun] = []
    removed: list[BatteryRun] = []
    for run in runs:
        if has_negative_current(run):
            removed.append(run)
        else:
            admitted.append(run)
    return admitted, removed


def sort_runs_chronologically(runs: Iterable[BatteryRun]) -> list[BatteryRun]:
    return sorted(runs, key=lambda run: run.timestamp)


def load_static_train_test_split(
    npz_dir: str | Path,
    train_count: int,
    date_fmt: str = DEFAULT_DATE_FORMAT,
    filter_negative_current: bool = True,
) -> StaticTrainTestSplit:
    dataset = BatteryRunDataset.from_directory(
        npz_dir=npz_dir,
        date_fmt=date_fmt,
        filter_negative_current=filter_negative_current,
    )
    return dataset.static_split(train_count)


def load_rolling_block(
    npz_dir: str | Path,
    file_names: Sequence[str],
    seq_len: int,
    date_fmt: str = DEFAULT_DATE_FORMAT,
    filter_negative_current: bool = True,
) -> RollingBlock:
    dataset = BatteryRunDataset.from_directory(
        npz_dir=npz_dir,
        date_fmt=date_fmt,
        filter_negative_current=filter_negative_current,
    )
    block = dataset.rolling_block(file_names)
    padded_runs = tuple(run.padded_or_truncated(seq_len) for run in block.runs)
    return RollingBlock(runs=padded_runs)
