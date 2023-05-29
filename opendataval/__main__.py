from pathlib import Path
from typing import Annotated, Any, Optional

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


# fmt: off
# ruff: noqa: E501 D103
class JobModel(pa.DataFrameModel):  # TODO errors with the nullable
    experiment_id: Series[int] = pa.Field(alias="Id", check_name=True, coerce=True, ge=0, unique=True)
    random_state: Optional[Series[int]] = pa.Field(alias="Random State", check_name=True, coerce=True, nullable=True, ignore_na=True)

    dataset: Series[str] = pa.Field(alias="Dataset", check_name=True, isin=set(Register.Datasets))
    cache_dir: Optional[Series[str]] = pa.Field(alias="Cache Directory", check_name=True, coerce=True, nullable=True)
    dataval: Series[str] = pa.Field(alias="Data Evaluator", check_name=True, isin=set(DataEvaluator.Evaluators))
    # dataval_kwargs: json
    model: Series[str] = pa.Field(alias="Model", check_name=True, isin=set(Model.Models))
    device: Optional[Series[str]] = pa.Field(alias="Device", check_name=True)  # TODO ensure valid type/device, probs with lambda
    # train_kwargs: Series[dict[str, object]]  = pa.Field()# TODO having some trouble with json

    train_count: Optional[Series[int]] = pa.Field(alias="Train", check_name=True, ge=0, coerce=True, default=0)
    valid_count: Optional[Series[int]] = pa.Field(alias="Valid", check_name=True, ge=0, coerce=True, default=0)
    test_count: Optional[Series[int]] = pa.Field(alias="Count", check_name=True, ge=0, coerce=True, default=0)

    noise_rate: Optional[Series[float]] = pa.Field(alias="Noise Rate", check_name=True, ge=0.0, le=1.0, coerce=True, default=0.0)
    # noise_kwargs: json

    metric: Series[str] = pa.Field(alias="Metric", check_name=True, isin=set(metrics_dict), nullable=True)

cli = typer.Typer()

@cli.command("csv", no_args_is_help=True)
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
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = pd.read_csv(file_)
    validated_jobs = JobModel.validate(jobs)

    for job_id in id_:
        row = validated_jobs[validated_jobs[JobModel.experiment_id] == job_id].iloc[0]
        run(row.to_dict(), output_dir)


def run(row: dict[str, Any], output_dir: Path):
    dataval = DataEvaluator.Evaluators[row[JobModel.dataval]]()  # **dataval_kwargs

    exper_med = ExperimentMediator.model_factory_setup(
        dataset_name=row[JobModel.dataset],
        cache_dir=row.get(JobModel.cache_dir),
        force_download=False,
        train_count=row.get(JobModel.train_count, 5),  #TODO set to 0
        valid_count=row.get(JobModel.valid_count, 5),
        test_count=row.get(JobModel.test_count, 5),
        add_noise_func=mix_labels,  # TODO only supports mix_labels currently
        noise_kwargs={'noise_rate': row.get(JobModel.noise_rate, 0)},
        random_state=row.get(JobModel.random_state, None),

        model_name=row[JobModel.model],
        train_kwargs={'epochs': 5, 'batch_size': 50},  # TODO allow customizable kwargs
        device=row.get(JobModel.device, "cuda" if torch.cuda.is_available() else "cpu"),
        metric_name=row.get(JobModel.metric, None),
    ).set_output_directory(output_dir).compute_data_values([dataval])

    # Runs all experiments available
    exper_med.evaluate(em.noisy_detection, save_output=True)
    exper_med.evaluate(em.save_dataval, save_output=True)
    exper_med.evaluate(em.discover_corrupted_sample, save_output=True)
    exper_med.evaluate(em.remove_high_low, include_train=True, save_output=True)
    exper_med.evaluate(em.increasing_bin_removal, include_train=True, save_output=True)


if __name__ == "__main__":
    cli()
