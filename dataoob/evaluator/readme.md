
## `ExperimentMediator`
`ExperimentMediator` is helps make a cohesive and controlled experiment. By injecting a model, data loader, and dataevaluators, it will train the models and facilitate additional experiments.
```python
expermed = ExperimentrMediator(loader, data_evaluators, model, train_kwargs, metric_name)
```
Metric name is one of predefined metrics in `evaluator`. # TODO


From here we can run experiments by passing in an experiment function `(DataEvaluator, DataFetcher) - > dict[str, Any]`. There are 5 found `exper_methods.py` with three being plotable. All returns include a pandas `DataFrame`.
```python
df = expermed.evaluate(noisy_detection)
df, figure = expermed.plot(discover_corrupted_sample)
```

We pass the arguments directly into the `eval_func`. Sometimes we need to pass in the train_kwargs. This also passes in `metric_name` for graphing purposes afterwards. To do so:
```python
df = expermed.evaluate(noisy_detection, include_trian=True)
```

## Alternate constructors.
Besides the intended constructor method, we can create an `ExperimentMediator` through other methods that may be easier to create all at once. This makes it easier to get started and set up the same experiment but with different ``DataEvaluator``.
```python
exper_med_partial = ExperimentMediator.partial_setup(
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
    metric_name=metric_name
)
exper_med = exper_med_partial(data_evaluators=[ChildEvaluator()])
```