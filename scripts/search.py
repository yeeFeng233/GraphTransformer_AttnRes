import argparse
import csv
import fcntl
import math
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import ray
import yaml
from lightning.pytorch.callbacks import Callback
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

warnings.filterwarnings("ignore", message="The 'repr' attribute with value False.*")
warnings.filterwarnings("ignore", message="The 'frozen' attribute with value True.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

PROJECT_NAME = "your_wandb_project"
ENTITY_NAME = "your_wandb_entity"
SEEDS = [43]

GPU_HEAVY_CONVS = {"GPSConv", "GPSAttnRes", "GRIT", "GRITAttnRes"}
RESERVED_CONFIG_KEYS = {
    "logger_type",
    "max_epochs",
    "data_root",
    "num_workers",
    "devices_per_trial",
    "cpus_per_trial",
    "search_config",
    "output_dir",
    "report_every_n_epochs",
    "early_stopping_patience",
    "artifact_root",
    "training_log_dir",
    "monitor_metric",
}


def create_stepped_values(min_value, max_value, step):
    if all(float(v).is_integer() for v in [min_value, max_value, step]):
        min_value = int(min_value)
        max_value = int(max_value)
        step = int(step)
        return list(range(min_value, max_value + 1, step))

    values = []
    current = float(min_value)
    while current <= float(max_value) + 1e-12:
        values.append(round(current, 10))
        current += float(step)
    return values


def parse_search_param(values):
    if isinstance(values, list):
        return tune.choice(values)

    if not isinstance(values, dict):
        return values

    if "values" in values:
        return tune.choice(values["values"])

    min_value = values.get("min")
    max_value = values.get("max")
    distribution = values.get("distribution")

    if distribution == "loguniform":
        return tune.loguniform(min_value, max_value)

    if distribution in {"q_uniform", "q_log_uniform_values"}:
        step = values.get("q")
        return tune.choice(create_stepped_values(min_value, max_value, step))

    if min_value is not None and max_value is not None:
        if isinstance(min_value, int) and isinstance(max_value, int):
            return tune.randint(min_value, max_value + 1)
        return tune.uniform(min_value, max_value)

    return values


def sanitize_model_config(config):
    return {k: v for k, v in config.items() if k not in RESERVED_CONFIG_KEYS}


def append_live_result(output_dir, task, config, metrics):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"live_search_{task}.csv"

    row = {**sanitize_model_config(config), **metrics}
    priority_cols = [
        "search_config",
        "conv_layer",
        "gnn_type",
        "val_mae",
        "val_loss",
        "test_mae",
        "test_loss",
    ]
    other_cols = [col for col in row.keys() if col not in priority_cols]
    fieldnames = [col for col in priority_cols if col in row] + sorted(other_cols)

    with open(csv_path, "a+", newline="", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        has_content = handle.read(1) != ""
        handle.seek(0, os.SEEK_END)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not has_content:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)



def format_float_for_filename(value):
    if value is None:
        return "none"
    try:
        return f"{float(value):.3g}".replace("+", "").replace("-", "m").replace(".", "p")
    except (TypeError, ValueError):
        return str(value).replace("/", "_").replace(" ", "_")


def get_trial_log_path(config, trial_id):
    log_dir = config.get("training_log_dir")
    if not log_dir:
        return None

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"{config.get('task', 'task')}_{trial_id}_"
        f"L{config.get('num_layers')}_H{config.get('hidden_dim')}_"
        f"B{config.get('batch_size')}_heads{config.get('gps_num_heads')}_"
        f"stride{config.get('attnres_history_stride')}_"
        f"lr{format_float_for_filename(config.get('lr'))}_"
        f"wd{format_float_for_filename(config.get('weight_decay'))}.log"
    )
    return log_dir / name


def append_trial_log(config, trial_id, line):
    log_path = get_trial_log_path(config, trial_id)
    if log_path is None:
        return
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


class TuneMetricsCallback(Callback):
    """Lightweight Lightning callback that reports intermediate validation metrics to Ray Tune."""

    def __init__(self, report_every_n_epochs: int, config=None, trial_id=None):
        super().__init__()
        self.report_every_n_epochs = max(1, int(report_every_n_epochs))
        self.config = config or {}
        self.trial_id = trial_id or "unknown"

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch + 1
        if epoch % self.report_every_n_epochs != 0:
            return

        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss")
        if val_loss is None:
            return

        payload = {
            "epoch": epoch,
            "val_loss": float(val_loss.detach().cpu()),
        }
        val_mae = metrics.get("val_mae")
        if val_mae is not None:
            payload["val_mae"] = float(val_mae.detach().cpu())

        train_loss = metrics.get("train_loss")
        if train_loss is not None:
            payload["train_loss"] = float(train_loss.detach().cpu())
        train_mae = metrics.get("train_mae")
        if train_mae is not None:
            payload["train_mae"] = float(train_mae.detach().cpu())

        cfg = getattr(pl_module, "hparams", {})
        line = (
            "[epoch_summary] "
            f"trial={self.trial_id} "
            f"epoch={epoch} "
            f"layers={getattr(cfg, 'num_layers', None)} "
            f"hidden={getattr(cfg, 'hidden_dim', None)} "
            f"heads={getattr(cfg, 'gps_num_heads', None)} "
            f"stride={getattr(cfg, 'attnres_history_stride', None)} "
            f"lr={getattr(cfg, 'lr', None)} "
            f"wd={getattr(cfg, 'weight_decay', None)} "
            + " ".join(f"{k}={v:.6g}" for k, v in payload.items() if isinstance(v, float))
        )
        append_trial_log(self.config, self.trial_id, line)

        session.report(payload)


def create_data_loaders(task, config):
    from torch_geometric.loader import DataLoader

    from utils import KHopTransform, get_dataset

    batch_size = config.get("batch_size")
    if batch_size is None:
        batch_size = 256 if config.get("conv_layer") in GPU_HEAVY_CONVS else 512

    pre_transform = None
    if config.get("gnn_type") == "DRew_GCN":
        pre_transform = KHopTransform(k=config.get("khop", 1))

    data_root = config.get("data_root", str(PROJECT_ROOT / "data"))
    data_train, data_val, data_test, num_feat, num_class = get_dataset(
        root=data_root,
        task=task,
        pre_transform=pre_transform,
        constant_feature=config.get("constant_feature"),
    )

    scaling_factor = data_train.scaling_factor[task]
    if scaling_factor is None and task in ["charge", "energy"]:
        scaling_factor = 1.0

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": config.get("num_workers", 2),
        "pin_memory": True,
    }

    train_loader = DataLoader(data_train, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(data_val, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(data_test, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, num_feat, num_class, scaling_factor


def create_model_and_trainer(config, num_feat, num_class, scaling_factor, task, logger_type="wandb", trial_id=None):
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, WandbLogger

    from utils.litmodels import LitGraphNN

    artifact_root = Path(config.get("artifact_root", PROJECT_ROOT))
    report_every_n_epochs = int(config.get("report_every_n_epochs", 5))
    patience = int(config.get("early_stopping_patience", 8))
    monitor_metric = config.get("monitor_metric", "val_loss")

    if logger_type == "wandb":
        logger = WandbLogger(
            log_model=False,
            project=PROJECT_NAME,
            save_dir=str(artifact_root / "logs" / "wandb" / task),
            entity=ENTITY_NAME,
        )
        experiment_name = logger.experiment.name
    else:
        logger = CSVLogger(
            save_dir=str(artifact_root / "logs" / "csv" / task),
            name="search",
        )
        experiment_name = f"version_{logger.version}"

    model = LitGraphNN(
        input_dim=num_feat,
        output_dim=num_class,
        node_level_task=task not in ["diam", "energy"],
        scaling_factor=scaling_factor or 1.0,
        edge_dim=2 if task in ["energy", "charge"] else None,
        **sanitize_model_config(config),
    )

    ckpt_callback = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath=str(artifact_root / "checkpoints" / task / config["conv_layer"] / experiment_name),
        save_top_k=1,
        filename="{epoch:02d}-{" + monitor_metric + ":.4f}",
        mode="min",
    )
    es_callback = EarlyStopping(monitor=monitor_metric, patience=patience, mode="min")

    devices_per_trial = int(config.get("devices_per_trial", 1))
    trainer = L.Trainer(
        max_epochs=config.get("max_epochs", 500),
        accelerator="gpu",
        devices=devices_per_trial,
        strategy="ddp_find_unused_parameters_true" if devices_per_trial > 1 else "auto",
        logger=logger,
        callbacks=[ckpt_callback, es_callback, TuneMetricsCallback(report_every_n_epochs, config=config, trial_id=trial_id)],
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=200,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=report_every_n_epochs,
    )
    return model, trainer


def train_model_tune(config):
    import lightning as L
    import torch
    import wandb

    torch.set_float32_matmul_precision("high")
    L.seed_everything(SEEDS[0])

    logger_type = config.get("logger_type", "wandb")
    if logger_type == "wandb":
        wandb.init(project=PROJECT_NAME, config=config, reinit=True, entity=ENTITY_NAME)

    trial_id = session.get_trial_id()
    task = config["task"]
    print(
        f"[{trial_id}] start task={task} layers={config.get('num_layers')} hidden={config.get('hidden_dim')} "
        f"batch={config.get('batch_size')} heads={config.get('gps_num_heads')} stride={config.get('attnres_history_stride')}"
    )

    train_loader, val_loader, test_loader, num_feat, num_class, scaling_factor = create_data_loaders(task, config)
    append_trial_log(config, trial_id, "[trial_start] " + " ".join(f"{k}={v}" for k, v in sorted(sanitize_model_config(config).items())))
    model, trainer = create_model_and_trainer(config, num_feat, num_class, scaling_factor, task, logger_type, trial_id=trial_id)

    trainer.fit(model, train_loader, val_loader)
    val_results = trainer.validate(model, val_loader, ckpt_path="best", verbose=False)
    test_results = trainer.test(model, test_loader, ckpt_path="best", verbose=False)

    final_metrics = {
        "epoch": trainer.current_epoch + 1,
        "val_loss": val_results[0]["val_loss"],
        "val_mae": val_results[0].get("val_mae"),
        "test_loss": test_results[0]["test_loss"],
        "test_mae": test_results[0].get("test_mae"),
    }

    append_live_result(config["output_dir"], task, config, final_metrics)
    done_line = (
        f"[trial_done] trial={trial_id} task={task} val_loss={final_metrics['val_loss']:.5f} "
        f"val_mae={final_metrics['val_mae']:.6f} test_mae={final_metrics['test_mae']:.6f}"
    )
    append_trial_log(config, trial_id, done_line)
    print(done_line)
    session.report(final_metrics)

    if logger_type == "wandb":
        wandb.finish()


def load_search_space(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    search_space = {}
    for param, values in config["parameters"].items():
        search_space[param] = parse_search_param(values)

    model_name = config_path.stem
    gnn_type = config.get("gnn_type")
    if gnn_type:
        search_space["gnn_type"] = gnn_type

    return search_space, model_name, config.get("initial_points")


def save_trial_results(trials, task, output_dir):
    csv_file = os.path.join(output_dir, f"search_{task}.csv")

    results = []
    for trial in trials:
        if not trial.last_result:
            continue
        row_data = {**trial.config, **trial.last_result}
        row_data["trial_id"] = getattr(trial, "trial_id", "")
        results.append(row_data)

    if not results:
        return

    df = pd.DataFrame(results)
    priority_cols = [
        "search_config",
        "conv_layer",
        "gnn_type",
        "val_mae",
        "val_loss",
        "test_mae",
        "test_loss",
    ]
    other_cols = [col for col in df.columns if col not in priority_cols]
    column_order = [col for col in priority_cols if col in df.columns] + other_cols
    df = df[column_order]
    df.to_csv(csv_file, index=False)


def get_config_files(task, model_names=None):
    config_dir = PROJECT_ROOT / "search-space" / task
    if not config_dir.exists():
        print(f"Warning: Config directory {config_dir} does not exist")
        return []

    yaml_files = list(config_dir.glob("*.yaml"))
    if model_names:
        yaml_files = [path for path in yaml_files if path.stem in model_names]
        if not yaml_files:
            print(f"No config files found for models {model_names} in task {task}")
    return sorted(yaml_files)


def create_scheduler(args):
    scheduler = None
    if args.scheduler == "asha":
        reports_per_trial = math.ceil(args.max_epochs / args.report_every_n_epochs) + 1
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            metric=args.monitor_metric,
            mode="min",
            max_t=reports_per_trial,
            grace_period=args.asha_grace_period or max(3, min(8, reports_per_trial // 4)),
            reduction_factor=2,
        )

    return scheduler


def create_search_alg(args, initial_points=None):
    if args.search_alg != "optuna":
        return None

    if initial_points:
        return OptunaSearch(metric=args.monitor_metric, mode="min", points_to_evaluate=initial_points)
    return OptunaSearch(metric=args.monitor_metric, mode="min")


def complete_initial_points(search_space, initial_points=None):
    if not initial_points:
        return initial_points

    completed = []
    for point in initial_points:
        full_point = dict(point)
        for key, value in search_space.items():
            if key in full_point or key in RESERVED_CONFIG_KEYS or key in {"task", "gnn_type"}:
                continue
            categories = getattr(value, "categories", None)
            if categories is not None and len(categories) == 1:
                full_point[key] = categories[0]
        completed.append(full_point)
    return completed


def run_experiments_for_task(task, args, scheduler):
    config_files = get_config_files(task, args.models)
    if not config_files:
        print(f"No config files found for task {task}")
        return

    all_trials = []
    for config_file in config_files:
        search_space, model_name, initial_points = load_search_space(config_file)
        print(f"Running {model_name} for task={task}")
        if initial_points:
            print(f"  Using {len(initial_points)} initial Optuna points from {config_file.name}")

        search_space["task"] = task
        search_space["logger_type"] = args.logger
        search_space["max_epochs"] = args.max_epochs
        search_space["data_root"] = str(PROJECT_ROOT / "data")
        search_space["num_workers"] = args.num_workers
        search_space["devices_per_trial"] = args.gpus_per_trial
        search_space["cpus_per_trial"] = args.cpus_per_trial
        search_space["search_config"] = config_file.stem
        search_space["output_dir"] = str(Path(args.output_dir).resolve())
        search_space["training_log_dir"] = str(Path(args.training_log_dir).resolve()) if args.training_log_dir else None
        search_space["monitor_metric"] = args.monitor_metric
        search_space["report_every_n_epochs"] = args.report_every_n_epochs
        search_space["early_stopping_patience"] = args.early_stopping_patience or max(
            4,
            math.ceil(args.max_epochs / args.report_every_n_epochs / 4),
        )
        search_space["artifact_root"] = str(PROJECT_ROOT)

        initial_points = complete_initial_points(search_space, initial_points)
        search_alg = create_search_alg(args, initial_points)
        analysis = tune.run(
            train_model_tune,
            config=search_space,
            num_samples=args.n_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            resources_per_trial={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
            name=f"{task}_{model_name}",
            storage_path=args.storage_path,
            verbose=0,
            raise_on_failed_trial=False,
        )
        all_trials.extend(analysis.trials)

    save_trial_results(all_trials, task, args.output_dir)
    print(f"Completed task={task}. Saved {len(all_trials)} trial records.")


def print_experiment_summary(tasks, output_dir):
    print("\nExperiment Summary:")
    for task in tasks:
        csv_file = os.path.join(output_dir, f"search_{task}.csv")
        if not os.path.exists(csv_file):
            continue

        df = pd.read_csv(csv_file)
        print(f"\nTask {task}:")
        print(f"  Total completed trials: {len(df)}")
        for metric in ("val_mae", "val_loss"):
            if metric in df.columns:
                valid = df[metric].dropna()
                if len(valid) > 0:
                    best_idx = df[metric].idxmin()
                    best = df.loc[best_idx]
                    print(
                        f"  Best by {metric}: {metric}={best[metric]:.6f}, "
                        f"val_loss={best.get('val_loss', float('nan')):.5f}, "
                        f"val_mae={best.get('val_mae', float('nan')):.6f}, "
                        f"test_mae={best.get('test_mae', float('nan')):.6f}"
                    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=16, help="Number of samples per search-space file")
    parser.add_argument("--tasks", nargs="+", default=["charge", "energy"], help="Tasks to run experiments on")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results CSV files")
    parser.add_argument("--storage_path", type=str, default="/tmp/ray_results", help="Ray Tune storage path")
    parser.add_argument("--ray_temp_dir", type=str, default=None, help="Ray session/temp directory")
    parser.add_argument("--num_cpus", type=int, default=None, help="Number of CPUs to expose to Ray")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to expose to Ray")
    parser.add_argument("--cpus_per_trial", type=int, default=4, help="CPU resources to reserve for each Ray trial")
    parser.add_argument("--gpus_per_trial", type=int, default=1, help="GPU resources to reserve for each Ray trial")
    parser.add_argument("--num_workers", type=int, default=2, help="PyG dataloader workers per trial")
    parser.add_argument("--models", nargs="+", default=None, help="Specific search-space files to test")
    parser.add_argument("--entity_name", type=str, default=ENTITY_NAME, help="Wandb entity name for logging")
    parser.add_argument("--scheduler", type=str, default="asha", choices=["asha", "none"], help="Scheduler to use")
    parser.add_argument("--search_alg", type=str, default="optuna", choices=["optuna", "random"], help="Search algorithm")
    parser.add_argument("--logger", type=str, default="csv", choices=["wandb", "csv"], help="Logger to use")
    parser.add_argument("--max_epochs", type=int, default=300, help="Maximum number of epochs for training")
    parser.add_argument("--report_every_n_epochs", type=int, default=5, help="Validation/report interval in epochs")
    parser.add_argument("--early_stopping_patience", type=int, default=None, help="Lightning early stopping patience in validation checks")
    parser.add_argument("--monitor_metric", type=str, default="val_loss", choices=["val_loss", "val_mae"], help="Metric for Optuna/ASHA/checkpoint/early stopping")
    parser.add_argument("--asha_grace_period", type=int, default=None, help="ASHA grace period in reported validation checks")
    parser.add_argument("--training_log_dir", type=str, default=None, help="Directory for per-trial epoch training logs")
    return parser.parse_args()


def main():
    args = parse_arguments()

    ray.init(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        _temp_dir=args.ray_temp_dir,
        runtime_env={
            "working_dir": str(PROJECT_ROOT),
            "excludes": [
                "data/",
                "checkpoints/",
                "logs/",
                "ray_tmp/",
                "ray_results/",
                "results/",
                ".git/",
                ".venv/",
                "__pycache__/",
                ".vscode/",
            ],
        },
    )
    os.makedirs(args.output_dir, exist_ok=True)

    scheduler = create_scheduler(args)
    print(
        f"Search start: tasks={args.tasks}, n_samples={args.n_samples}, max_epochs={args.max_epochs}, "
        f"report_every_n_epochs={args.report_every_n_epochs}, cpus_per_trial={args.cpus_per_trial}, "
        f"gpus_per_trial={args.gpus_per_trial}"
    )

    for task in args.tasks:
        run_experiments_for_task(task, args, scheduler)

    print_experiment_summary(args.tasks, args.output_dir)


if __name__ == "__main__":
    main()
    ray.shutdown()
