## `DataEvaluator`
Provides an Abstract Base Class to evaluate data values.
The input data points must be a tensor. An example class is down below:
```python
from dataoob.dataval import DataEvaluator

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
    .input_model_metric(model, metric)
    .input_data(x_train, y_train, x_valid, y_valid)
    .train_data_values()
)

data_values = dataval.evaluate_data_values()
```

## `ShapEvaluator`
A number of DataEvaluators are actually semivalues, which means we can reuse the marginal computations of across several evaluators. To do so: #TODO