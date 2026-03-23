from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    mode: str
    V0: float
    x0: tuple[float, float, float]
    dt: float
    VEOD: float
    Rp: float
    Rs: float
    Csp: float
    Cs: float


@dataclass(frozen=True)
class DatasetConfig:
    npz_dir: str
    train_runs: int
    seq_len: int


@dataclass(frozen=True)
class TrainingConfig:
    device: str
    epoch: int
    lr: float
    weight_decay: float
    plot_n: int = 3
    mc_samples: int = 100
    n_jobs: int = 6
    kl_beta: float = 1e-5


@dataclass(frozen=True)
class RollingWindowConfig:
    block_size: int = 10
    save_dir: str = './models'


@dataclass(frozen=True)
class OutputConfig:
    results_dir: str = 'results'
    results_prefix: str = 'BattNN'
    save_model: str = 'NASA'


@dataclass(frozen=True)
class BatteryExperimentConfig:
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    output: OutputConfig
    rolling: RollingWindowConfig | None = None

    def to_namespace(self):
        payload = {
            'model_name': self.model.model_name,
            'mode': self.model.mode,
            'V0': self.model.V0,
            'x0': list(self.model.x0),
            'dt': self.model.dt,
            'VEOD': self.model.VEOD,
            'Rp': self.model.Rp,
            'Rs': self.model.Rs,
            'Csp': self.model.Csp,
            'Cs': self.model.Cs,
            'npz_dir': self.dataset.npz_dir,
            'train_runs': self.dataset.train_runs,
            'batch_size': self.dataset.train_runs,
            'seq_len': self.dataset.seq_len,
            'device': self.training.device,
            'epoch': self.training.epoch,
            'lr': self.training.lr,
            'weight_decay': self.training.weight_decay,
            'plot_n': self.training.plot_n,
            'mc_samples': self.training.mc_samples,
            'n_jobs': self.training.n_jobs,
            'kl_beta': self.training.kl_beta,
            'results_dir': self.output.results_dir,
            'results_prefix': self.output.results_prefix,
            'save_model': self.output.save_model,
        }
        if self.rolling is not None:
            payload['block_size'] = self.rolling.block_size
            payload['save_dir'] = self.rolling.save_dir
        return SimpleNamespace(**payload)


def build_static_experiment_config(args) -> BatteryExperimentConfig:
    return BatteryExperimentConfig(
        model=ModelConfig(
            model_name=args.model_name,
            mode=args.mode,
            V0=args.V0,
            x0=tuple(args.x0),
            dt=args.dt,
            VEOD=args.VEOD,
            Rp=args.Rp,
            Rs=args.Rs,
            Csp=args.Csp,
            Cs=args.Cs,
        ),
        dataset=DatasetConfig(
            npz_dir=args.npz_dir,
            train_runs=args.train_runs,
            seq_len=args.seq_len,
        ),
        training=TrainingConfig(
            device=args.device,
            epoch=args.epoch,
            lr=args.lr,
            weight_decay=args.weight_decay,
            plot_n=args.plot_n,
            mc_samples=args.mc_samples,
            n_jobs=args.n_jobs,
            kl_beta=getattr(args, 'kl_beta', 1e-5),
        ),
        output=OutputConfig(
            results_dir=args.results_dir,
            results_prefix=args.results_prefix,
            save_model=args.save_model,
        ),
    )


def build_rolling_experiment_config(args, mode: str) -> BatteryExperimentConfig:
    return BatteryExperimentConfig(
        model=ModelConfig(
            model_name=args.model_name,
            mode=mode,
            V0=args.V0,
            x0=tuple(args.x0),
            dt=args.dt,
            VEOD=args.VEOD,
            Rp=args.Rp,
            Rs=args.Rs,
            Csp=args.Csp,
            Cs=args.Cs,
        ),
        dataset=DatasetConfig(
            npz_dir=args.npz_dir,
            train_runs=args.train_runs,
            seq_len=args.seq_len,
        ),
        training=TrainingConfig(
            device=args.device,
            epoch=args.epoch,
            lr=args.lr,
            weight_decay=args.weight_decay,
            mc_samples=args.mc_samples,
            n_jobs=getattr(args, 'n_jobs', 6),
            kl_beta=getattr(args, 'kl_beta', 1e-5),
        ),
        output=OutputConfig(
            results_dir='results',
            results_prefix='RollingVBattNN' if mode == 'variational' else 'RollingBattNN',
            save_model=args.save_model,
        ),
        rolling=RollingWindowConfig(
            block_size=args.block_size,
            save_dir=args.save_dir,
        ),
    )
