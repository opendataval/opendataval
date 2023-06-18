import json
import warnings
from pathlib import Path
from typing import Annotated, Any, Optional

import numpy as np
import pandas as pd
import pandera as pa
import typer
from pandera.typing import Series
from typer import Option

from opendataval.dataloader import NoiseFunc, Register
from opendataval.dataval import DataEvaluator
from opendataval.experiment import ExperimentMediator
from opendataval.experiment import exper_methods as em
from opendataval.metrics import Metrics
from opendataval.model import Model
from opendataval.util import StrEnum

# fmt: off
# ruff: noqa: E501 D103

# Enums for better types, used with typer for better CLI
DatasetsEnum = StrEnum("Datasets", list(Register.Datasets))
DataEvaluatorsEnum = StrEnum("DataEvaluators", list(DataEvaluator.Evaluators))
ModelsEnum = StrEnum("Models", list(Model.Models))

def _json_loads(x: str) -> dict[str, Any]:
    """Loads json, returns empty on failure."""
    if isinstance(x, dict):
        return x

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

    add_noise: Optional[str] = pa.Field(alias="Add Noise", check_name=True, isin=set(NoiseFunc))  # TODO fix
    noise_kwargs: Series[object] = pa.Field(alias="Noise Arguments", check_name=True, nullable=True)

    dataval: Series[str] = pa.Field(alias="Data Evaluator", check_name=True, isin=set(DataEvaluator.Evaluators))
    dataval_kwargs: Series[object] = pa.Field(alias="Data Valuation Arguments", check_name=True, nullable=True)

    model: Series[str] = pa.Field(alias="Model", check_name=True, isin=set(Model.Models))
    device: Optional[Series[str]] = pa.Field(alias="Device", check_name=True)
    train_kwargs: Series[object] = pa.Field(alias="Training Arguments", check_name=True, nullable=True)

    metric: Series[str] = pa.Field(alias="Metric", check_name=True, isin=Metrics, nullable=True)

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

@cli.command("filerun")
def setup(
    file_: Annotated[typer.FileText, Option("--file", "-f", help="CSV file containing jobs", prompt=True)],
    id_: Annotated[list[int], Option("--id", "-n", help="Id of the job", prompt=True)],
    output_dir: Annotated[Optional[Path], Option(
            "--output", "-o", help="Directory of experiments output",
            dir_okay=True, writable=True, resolve_path=True, prompt=True
    )] = ".",
):
    """CLI run a singular job from an input CSV file

    Parameters
    ----------
    file_ : typer.FileText
        File path containing jobs to be run. Must be in the specified job format
    id_ : list[int]
        IDs of job to be run, called with multiple ``-n`` arguments
    output_dir : Path, optional
        Directory of outputs of the experiment. Must be a possible directory, by default
        Path(".") or current working directory.
    """
    jobs = pd.read_csv(file_)
    validated_jobs = JobModel.validate(jobs)

    for job_id in id_:
        row = validated_jobs[validated_jobs[JobModel.experiment_id] == job_id].iloc[0]
        typer.echo("Current job {job_id} is as follows")
        typer.echo(row)
        run(
            dataval=row[JobModel.dataval],
            dataset=row[JobModel.dataset],
            model=row[JobModel.model],
            add_noise=row.get(JobModel.add_noise),
            metric=row.get(JobModel.metric),
            dataval_kwargs=row.get(JobModel.dataval_kwargs),
            train_kwargs=row.get(JobModel.train_kwargs),
            noise_kwargs=row.get(JobModel.noise_kwargs),
            cache_dir=row.get(JobModel.cache_dir),
            train=int(row.get(JobModel.train_count, 25)),  # TODO remove force cast to int
            valid=int(row.get(JobModel.valid_count, 25)),
            test=int(row.get(JobModel.test_count, 25)),
            device=row.get(JobModel.device),
            random_state=row.get(JobModel.random_state),
            output_dir=output_dir / job_id
        )

@cli.command("run")
def run(
    dataval: Annotated[DataEvaluatorsEnum, Option(help="Data Evaluator Name", prompt=True)],
    dataset: Annotated[DatasetsEnum, Option(help="Dataset Name", prompt=True)],
    model: Annotated[ModelsEnum, Option(help="Model Name", prompt=True)],
    add_noise: Annotated[Optional[NoiseFunc], Option(help="Adding artificial noise method")] = None,
    metric: Annotated[Optional[Metrics], Option(help="Metric of evaluating the prediction model")] = None,

    dataval_kwargs: Annotated[Optional[dict[str, Any]], Option(help="Data Evaluator Key Work Arguments", parser=_json_loads, prompt=True)] = None,
    noise_kwargs: Annotated[Optional[dict[str, Any]], Option(help="Adding noise Key Work Arguments", parser=_json_loads)] = None,
    train_kwargs: Annotated[Optional[dict[str, Any]], Option(help="Model training Key Work Arguments", parser=_json_loads, prompt=True)] = None,

    cache_dir: Annotated[Optional[Path], Option(
        help="Directory to cache downloads",
        dir_okay=True, writable=True, resolve_path=True,)] = None,
    train: Annotated[Optional[int], Option(
        help="Number of training data points", prompt=True)] = 25,
    valid: Annotated[Optional[int], Option(
        help="Number of validation data points", prompt=True)] = 25,
    test: Annotated[Optional[int], Option(
        help="Number of test data points", prompt=True)] = 25,

    device: Annotated[Optional[str], Option(help="Torch device")] = None,
    random_state: Annotated[Optional[int], Option(help="Initial Random State")] = None,
    output_dir: Annotated[Optional[Path], Option(
        "--output", "-o", help="Directory of experiments output",
        prompt=True, dir_okay=True, writable=True, resolve_path=True)] = ".",
):
    """CLI to run an opendataval job form input parameters

    Parameters
    ----------
    dataval : DataEvaluatorsEnum, str
        Data evaluator name, must match class name exactly
    dataset : DatasetsEnum, str
        Name of the data set, must match registered data set exactly
    model : ModelsEnum, str
        Name of the model, should match class name, see py:function`~opendataval.model.ModelFactory`
    add_noise : NoiseFunc, str, optional
        Method of adding artificial noise, must match names in py:class`~opendataval.dataloader.NoiseFunc`
    metric : Metrics, str, optional
        Method of evaluating the prediction model, must match names in py:class`~opendataval.metrics.Metrics`
    dataval_kwargs : dict[str, Any], optional, str
        Data Evaluator Key Work Arguments, by default {}
    noise_kwargs : dict[str, Any], optional
        Adding noise Key Work Arguments, by default {}
    train_kwargs : dict[str, Any], optional
        Model training Key Work Arguments, by default {}
    cache_dir : Path, optional
        Directory to cache downloads, by default Path("data_files/")
    train : int, optional
        Number of training data points, by default 25
    valid : int, optional
        Number of validation data points, by default 25
    test : int, optional
        Number of test data points, by default 25
    device : str, optional
        Torch device, by default CPU
    random_state : int, optional
         Initial Random State, by default None
    output_dir : Path, optional
        Directory of experiments output, by default current working directory
    """
    dataval = DataEvaluator.Evaluators[dataval](**dataval_kwargs)  # **dataval_kwargs

    typer.echo(f"Starting computation of data values for {dataval=}")
    exper_med = ExperimentMediator.model_factory_setup(
        dataset_name=dataset,
        cache_dir=cache_dir,
        force_download=False,
        train_count=train,
        valid_count=valid,
        test_count=test,
        add_noise=add_noise,
        noise_kwargs=noise_kwargs,
        random_state=random_state,
        model_name=model,
        train_kwargs=train_kwargs,
        device=device,
        metric_name=metric,
        output_dir=output_dir,
    ).compute_data_values([dataval])
    typer.echo(f"Completed computation of data values for {dataval=}")

    # Runs all experiments available
    exper_med.evaluate(em.save_dataval, save_output=True)
    exper_med.evaluate(em.noisy_detection, save_output=True)
    exper_med.evaluate(em.discover_corrupted_sample, save_output=True)
    exper_med.evaluate(em.remove_high_low, include_train=True, save_output=True)
    exper_med.evaluate(em.increasing_bin_removal, include_train=True, save_output=True)

if __name__ == "__main__":
    cli()
