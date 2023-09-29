### `Model`
`Model` is an abstract base class that requires the implementation of three methods: `.train(x, y)`, `.predict(x)`, `.clone()`.

`.predict(x)` should return a tensor with the same dimensionality, not same length, as the input label `y` into `.train(x, y)`
```python
y_hat = child_model.fit(x, y).predict(x_valid)
assert y_hat.shape == y_valid.shape
```
`Model` was primarily designed with PyTorch models in mind, which is why there are additional mixins to inherit for PyTorch models. The mixins implement the fit and predict functions for you. So long as the child class is a valid nn.Module.
```python
from opendataval.model import TorchPredictMixin, TorchClassMixin
class TorchNN(TorchPredictMixin, TorchClassMixin):
    def __init__(self, *args, **kwargs):
        ...
    def forward(self, x):
        return x
```

There is also support for sk-learn models through a wrapper. In edge cases where we don't have all the labels due small subset, the model will be replaced with a naive dummy model, representing how the model isn't a valid predictor:
```python
from opendataval.model import ClassifierSkLearnWrapper
from sklearn.linear_model import LogisticRegression

wrapped_lr = ClassifierSkLearnWrapper(LogisticRegression(), label_dim, device=torch.device('...'))

wrapped_lr.fit(x, y)  # Accepts only tensor inputs and converts them to numpy
wrapped_lr.predict(x)  # Equivalent of predict_proba(x) on classification, .predict() for regression. Returns tensor
```

There is also a `ModelFactory` to create models from a name. A fetcher
will specify the input and output dimensions.
```python
from opendataval.dataloader import DataFetcher
from opendataval.model import ModelFactory

fetcher = DataFetcher('datset_name')  # Fetcher has covar/label dim information
mlp_classifier = ModelFactory('ClassifierMLP', fetcher, torch.device(...), *args, **kwargs)
```