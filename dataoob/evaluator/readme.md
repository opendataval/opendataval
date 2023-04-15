
## `ExperimentMediator`
`ExperimentMediator` is helps make a cohesive and controlled experiment. By injecting a model, data loader, and dataevaluators, it will train the models and facilitate additional experiments.
```python
expermed = ExperimentrMediator(
    loader, data_evaluators, model, train_kwargs, metric_name
)
```
Metric name is one of predefined metrics in `evaluator`. # TODO


From here we can run experiments by passing in an experiment function `(DataEvaluator, DataLoader) - > dict[str, Any]`. There are 5 found `exper_methods.py` with three being plotable. All returns include a pandas `DataFrame`.
```python
df = expermed.evaluate(noisy_detection)
df, figure = expermed.plot(discover_corrupted_sample)
```

We pass the arguments directly into the `eval_func`. Sometimes we need to pass in the train_kwargs. This also passes in `metric_name` for graphing purposes afterwards. To do so:
```python
df = expermed.evaluate(noisy_detection, include_trian=True)
```

## Alternate constructors.

We have two dataclasses that wrap the arguments so we can define exactly what we want to input into an `ExperimentMediator` prior. This is a quality of life feature to make it easier to understand the inputs to `ExperimentMediator`

For example:
```python
from dataoob.evaluator import DataEvaluatorArgs, DataLoaderArgs, ExperimentMediator

loader_args = DataLoaderArgs(
    dataset="dataset_name",
    force_download=True,
    device=torch.device("cuda"),
    train_count=.7,
    valid_count=0.2,
    test_count=0.1,
    noise_kwargs={'noise_rate': .2}
    add_noise_func=mix_labels  # (DataLoader, ...) -> dict[str, np.ndarray]
)

dataval_args = DataEvaluatorArgs(
    pred_mode=ANN(4, 3),
    train_kwargs={'epochs': 32, 'batch_size': 20},
    metric_name='accuracy'
)

exper_med = ExperimentMediator.from_dataclasses(loader_args, dataval_args)
```

## `presets.py`
We define many presets to quickly get an `ExperimentMediator` and test `exper_methods.py`. To use call one of the following functions: `new_evaluator` if we're testing a new `DataEvaluator` with `DataLoader` `Model` presets or `from_presets` if you want to just specify strings to get the `DataEvaluator` as well.
```python
from dataoob.evaluator import from_presets, new_evaluator
exper_med = from_presets('iris_ann_low_noise', 'dummy')
df = exper_med.evaluate(noisy_detection)

```