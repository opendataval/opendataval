## `DataEvaluator`
Provides an Abstract Base Class to evaluate data values.
The input data points must be a tensor. An example class is down below:
```python
from opendataval.dataval import DataEvaluator

class ChildEvaluator(DataEvaluator):
    def __init__(*args, **kwargs):
        ...  # Implementation and args are up to you

    def input_data(
        self,
        x_train: torch.Tensor | Dataset,
        y_train: torch.Tensor,
        x_valid: torch.Tensor | Dataset,
        y_valid: torch.Tensor,
    ) -> Self:
        """Input the data and set up some data dependent configs"""
        ...

    def train_data_values(self, *train_args, **train_kwargs) -> Self:
        """Takes no arguments, the inputs are passed to the model. Multiple calls should train the model more, but not a guarantee for all models"""
        ...

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point"""
        ...
```

To inject model, metrics, and data points for training
```python
dataval = (
    dataval
    .input_model(model)
    .input_metric(metric)
    .input_data(x_train, y_train, x_valid, y_valid)
    .train_data_values()
)

data_values = dataval.data_values  # Cached
data_values = dataval.evaluate_data_values()
```

For a short cut, say we have the evaluator and just want to input the data and model.
The model will infer a default evaluation metric based on the type of data specified by
fetcher. fetcher is a `DataFetcher` and model is a `Model`
```python
data_values = ChildEvaluator.train(fetcher, model, *train_args, **train_kwargs).evaluate_data_values()
```


## `ShapEvaluator`
A number of DataEvaluators are actually semivalues, which means we can reuse the marginal computations of across several evaluators. To do so specify a `cache_name: str`.