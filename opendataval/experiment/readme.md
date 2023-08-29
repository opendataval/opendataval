
## `ExperimentMediator`
`ExperimentMediator` is helps make a cohesive and controlled experiment. By injecting a model, data loader, and dataevaluators, it will train the models and facilitate additional experiments.
NOTE Warnings are raised if errors occur in a specific `DataEvaluator`.
```python
expermed = ExperimentrMediator(loader, model, train_kwargs, metric_name).compute_data_values(data_evaluators)
```
Metric name is one of predefined metrics in `opendataval/metrics.py`.


From here we can run experiments by passing in an experiment function `(DataEvaluator, DataFetcher) - > dict[str, Any]`. There are 5 found `exper_methods.py` with three being plotable. All returns include a pandas `DataFrame`.
```python
df = expermed.evaluate(noisy_detection)
df, figure = expermed.plot(discover_corrupted_sample)
```

We pass the arguments directly into the `eval_func`. Sometimes we need to pass in the train_kwargs. This also passes in `metric_name` for graphing purposes afterwards. To do so:
```python
df = expermed.evaluate(noisy_detection)
```

## Alternate constructors.
Besides the intended constructor method, we can create an `ExperimentMediator` through a ModelFactory of defaults
```python
exper_med = ExperimentMediator.model_factory_setup(
    dataset_name=dataset_name,
    force_download=False,
    train_count=train_count,
    valid_count=valid_count,
    test_count=test_count,
    noise_kwargs=noise_kwargs,
    random_state=random_state,
    model_name=model_name,
    device=device,
    train_kwargs=train_kwargs,
    metric_name=metric_name,
    output_dir=output_dir,
)
exper_med = exper_med.compute_data_values(data_evaluators=[ChildEvaluator()])
```