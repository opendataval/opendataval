## `DataFetcher`
The DataFetcher is a class that will load exactly one data set per instance. It accepts the name of a `Register` data set and handles the preprocessing involved. For our purposes, we can find the registered datasets with:
```python
DataFetcher.datasets_available()  # ['name1', 'name2']
```

A fetcher first takes a data set name to be loaded.
```python
from dataoob.dataloader import DataFetcher

fetcher = DataFetcher(dataset_name='dataset')
```

From there we must define how we will split the data set into train/valid/test splits
```python
fetcher = fetcher.split_dataset(70, 20, 10)  # Data set counts
fetcher = fetcher.split_dataset(.7, .2, .1)  # Splits on proportions
```

Finally we can specify a function on how to add noise to the data points. The function should be allowed to access every instance variable of a data fetcher.
```
fetcher = fetcher.noisify(noise_func, **noise_kwargs)  # noise_func: (DataFetcher, ...) -> dict[str, np.ndarray]
```

The return type of this function is a dict with the following strings and the updated `np.ndarray`. If the key is present, we will update the value of `self.[key]` with the value in the dictionary.
```python
{'x_train','y_train','x_valid','y_valid','x_test','y_test','noisy_train_indices'}
```

To get data points
```python
x_train, y_train, x_valid, y_valid, x_test, y_test = fetcher.datapoints
```

Alternatively, if you're unhappy with the above implementation, we present
another way of constructing the DataFetcher by passing all arguments in at once:
```python
fetcher = DataFetcher.setup(
    dataset_name=dataset_name,
    force_download=force_download,
    random_state=random_state,
    train_count=train_count,
    valid_count=valid_count,
    test_counttest_count=test_count,
    add_noise_func=add_noise_func,
    noise_kwargs=noise_kwargs
)
x_train, y_train, x_valid, y_valid, x_test, y_test = fetcher.datapoints
```

## `Register` datasets
Data sets are tricky topic, as sometimes we'd want to load the covariates and labels seperately. If you're not contributing, please ignore this as this is all facaded away by DataFetcher.

Take for example this use case, we want to have the covariates loaded dynamically (like a PyTorch `Dataset`) because of the memory usage, but we need the labels all loaded in memory (for certain evaluators).

```python
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    ...

def image_labels() -> np.ndarray:
    ...
```

To ensure that these two separate functions are fetched together, we define a `Register` object to link these two.

```python
image_dataset = Register('image', categorical=True, cacheable=True)

@image_dataset.from_covar_func
class ImageDataset(Dataset):
    ...
@image_dataset.from_label_func
def image_labels() -> np.ndarray:
    ...
```

The typical way is to have a callable return a tuple of `(covariates, labels)`. If
that's the case the api is as follows
```python
@Register('name_of_dataset')
def covariate_label_func(...)
    ...

@Register('name_of_other_dataset').from_covar_label_func  # More explicitly
def other_covariate_label_func(...)
    ...

```

We also define transformations whenever, the
transform is called when the data is loaded.
```python
image_dataset.add_label_transform(one_hot_encode)

covar, labels = image_dataset.load_data()
# labels is not a one_hot_encoding
```

Finally, we can always register a data set from a pandas DataFrame, a csv file, or a numpy array
```python
pd_dataset = Register("pd").from_pandas(df, label_columns=['label1', 'label2'])
covar, labels = pd_dataset.load_data()
```

To wrap back around, this is the api to get the data from any of the Registered data sets.
```python
datapoints = DataFetcher("pd").split_dataset(.7, .2, .1).datapoints
```
