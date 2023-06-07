import json
import warnings
from pathlib import Path
from typing import Annotated, Any, Optional

import numpy as np
import pandas as pd
import pandera as pa
import torch
import typer
from pandera.typing import Series

from opendataval.dataloader import Register, mix_labels
from opendataval.dataval import DataEvaluator
from opendataval.experiment import exper_methods as em
from opendataval.experiment.api import ExperimentMediator, metrics_dict
from opendataval.model import Model
from opendataval.util import StrEnum

# fmt: off
# ruff: noqa: E501 D103

def _json_loads(x: str) -> dict[str, Any]:
    """Loads json, returns empty on failure."""
    try:
        return json.loads(x)
    except (ValueError, KeyError):
        warnings.warn("Invalid json, using empty dict")
        return {}

class JobModel(pa.DataFrameModel):  # TODO errors with the nullable
    experiment_id: Series[int] = pa.Field(alias="Id", check_name=True, coerce=True, ge=0, unique=True)
    random_state: Optional[Series[int]] = pa.Field(alias="Random State", check_name=True, coerce=True, nullable=True, ignore_na=True)

    dataset: Series[str] = pa.Field(alias="Dataset", check_name=True, isin=set(Register.Datasets))
    cache_dir: Optional[Series[str]] = pa.Field(alias="Cache Directory", check_name=True, coerce=True, nullable=True)

    train_count: Optional[Series[int]] = pa.Field(alias="Train", check_name=True, ge=0, coerce=True, default=0)
    valid_count: Optional[Series[int]] = pa.Field(alias="Valid", check_name=True, ge=0, coerce=True, default=0)
    test_count: Optional[Series[int]] = pa.Field(alias="Test", check_name=True, ge=0, coerce=True, default=0)

    noise_rate: Optional[Series[float]] = pa.Field(alias="Noise Rate", check_name=True, ge=0.0, le=1.0, coerce=True, default=0.0)
    noise_kwargs: Series[object] = pa.Field(alias="Noise Arguments", check_name=True, nullable=True)

    dataval: Series[str] = pa.Field(alias="Data Evaluator", check_name=True, isin=set(DataEvaluator.Evaluators))
    dataval_kwargs: Series[object] = pa.Field(alias="Data Valuation Arguments", check_name=True, nullable=True)

    model: Series[str] = pa.Field(alias="Model", check_name=True, isin=set(Model.Models))
    device: Optional[Series[str]] = pa.Field(alias="Device", check_name=True)
    train_kwargs: Series[object] = pa.Field(alias="Training Arguments", check_name=True, nullable=True)

    metric: Series[str] = pa.Field(alias="Metric", check_name=True, isin=set(metrics_dict), nullable=True)

    @classmethod
    def validate(cls, check_obj: pd.DataFrame, *args, **kwargs):
        """Validates _kwargs inputs can be casted to a dict."""
        vectorized_load = np.vectorize(_json_loads)
        for _, field in filter(lambda item: '_kwargs' in item[0], cls._get_model_attrs().items()):
            if field.alias in check_obj.columns:
                check_obj[field.alias] = vectorized_load(check_obj[field.alias])

        return super().validate(check_obj, *args, **kwargs)

cli = typer.Typer()
"""Typer CLI entry point."""

# Enums for better types, used with typer for better CLI
DatasetsEnum = StrEnum("Datasets", list(Register.Datasets))
DataEvaluatorsEnum = StrEnum("DataEvaluators", list(DataEvaluator.Evaluators))
ModelsEnum = StrEnum("Models", list(Model.Models))
MetricEnum = StrEnum("Metrics", list(metrics_dict))

@cli.command("filerun", no_args_is_help=True)
def setup(
    file_: Annotated[typer.FileText, typer.Option("--file", "-f", help="CSV file containing jobs")],
    id_: Annotated[list[int], typer.Option("--id", "-n", help="Id of the job")],
    output_dir: Annotated[Optional[Path], typer.Option(
            "--output",
            "-o",
            help="Directory of experiments output",
            dir_okay=True,
            writable=True,
            resolve_path=True,
        )
    ] = ".",
):
    """CLI input to run a singular job from an input CSV file

    Parameters
    ----------
    file_ : typer.FileText
        File path containing jobs to be run. Must be in the specified job format
    id_ : list[int]
        IDs of job to be run, called with multiple ``-n`` arguments
    output_dir : Optional[Path]
        Directory of outputs of the experiment. Must be a possible directory, by default
        Path(".") or current working directory.
    """
    jobs = pd.read_csv(file_)
    validated_jobs = JobModel.validate(jobs)

    for job_id in id_:
        row = validated_jobs[validated_jobs[JobModel.experiment_id] == job_id].iloc[0]
        run(row.to_dict(), job_id, output_dir)

def run(row: dict[str, Any], run_id: int, output_dir: Path):
    dataval = DataEvaluator.Evaluators[row[JobModel.dataval]](**row.get(JobModel.dataval_kwargs, {}))  # **dataval_kwargs

    typer.echo(f"Starting computation of data values for id={run_id}")
    exper_med = ExperimentMediator.model_factory_setup(
        dataset_name=row[JobModel.dataset],
        cache_dir=row.get(JobModel.cache_dir),
        force_download=False,
        train_count=row.get(JobModel.train_count, 25),  #TODO set to 0
        valid_count=row.get(JobModel.valid_count, 25),
        test_count=row.get(JobModel.test_count, 25),
        add_noise=mix_labels,  # TODO only supports mix_labels currently
        noise_kwargs=row.get(JobModel.noise_kwargs, None),
        random_state=row.get(JobModel.random_state, None),
        model_name=row[JobModel.model],
        train_kwargs=row.get(JobModel.train_kwargs, None),
        device=row.get(JobModel.device, "cuda" if torch.cuda.is_available() else "cpu"),
        metric_name=row.get(JobModel.metric, None),
        output_dir=output_dir / f"id={run_id}/",
    ).compute_data_values([dataval])
    typer.echo(f"Completed computation of data values for id={run_id}")

    # Runs all experiments available
    typer.echo(f"Starting experiment for id={run_id}")
    exper_med.evaluate(em.noisy_detection, save_output=True)
    exper_med.evaluate(em.save_dataval, save_output=True)
    exper_med.evaluate(em.discover_corrupted_sample, save_output=True)
    exper_med.evaluate(em.remove_high_low, include_train=True, save_output=True)
    exper_med.evaluate(em.increasing_bin_removal, include_train=True, save_output=True)
    typer.echo(f"Completed experiment for id={run_id}")


if __name__ == "__main__":
    cli()
