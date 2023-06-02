<!-- Here's a blank template to get started: To avoid retyping too much info. Do a search and replace with your text editor for the following: `kevinfjiang`, `opendataval` `kevinfjiang` -->

<a name="readme-top"></a>

<!-- PROJECT LOGO -->

<div width="175" align="right">
<a href="https://github.com/kevinfjiang/opendataval">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo-dark-mode.png">
      <source media="(prefers-color-scheme: light)" srcset="docs/_static/logo-light-mode.png">
      <img alt="Logo toggles light and dark mode" src="docs/_static/logo-light-mode.png"  width="300" align="right">
    </picture>
</a>
</div>

# OpenDataVal

> A unified library for  transparent data valuation benchmarks

[**Website**](opendataval.github.io) | [**arXiv Paper**](TODO)

Assessing the quality of individual data points is critical for improving model performance and mitigating biases. However, there is no way to systematically benchmark different algorithims.

**opendataval** is an open-source initiative that with a diverse array of datasets/models (image, NLP, and tabular), data valuation algorithims, and evaluation taks using just a few lines of code.


| Overview | |
|----------|-|
|**CI/CD**|[![Build][test-shield]][test-url] ![Coverage][coverage_badge] |
|**Python**|[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue?style=for-the-badge)](https://www.python.org/)|
|**Dependencies**|[![Pytorch][PyTorch-shield]][PyTorch-url] [![scikit-learn][scikit-learn-shield]][scikit-learn-url] [![numpy][numpy-shield]][numpy-url] [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge&logo=appveyor)](https://github.com/psf/black) |
|**Documentation**| [Documentation](opendataval.github.io) |
|**Issues**| [![Issues][issues-shield]][issues-url] |
|**PyPi**|[![Releases][release-shield]][release-url]|
|**License**|[![MIT License][license-shield]][license-url]|
|**Contributors**|[![Contributors][contributors-shield]][contributors-url]|
|**Citation**| TODO |

<!-- TABLE OF CONTENTS -->
<details>
  <summary>👇Table of Contents</summary>
  <hr>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#sparkles-features">Features</a></li>
    <li>
      <a href="#zap-quick-start">Quick Start</a>
      <ul>
        <li><a href="#computer-cli">CLI</a></li>
        <li><a href="#control_knobs-api">API</a></li>
      </ul>
    </li>
    <li><a href="#medal_sports-opendataval-leaderboards">Leaderboards</a></li>
    <li><a href="#wave-contributing">Contributing</a></li>
    <li><a href="#bulb-vision">Vision</a></li>
    <li><a href="#classical_building-license">License</a></li>
    <li><a href="#books-cite-us">Cite Us</a></li>
  </ol>
</details>



## :sparkles: Features
----

| Feature | Status | Links | Notes |
|---------|--------|-------|-------|
| **[Datasets](https://github.com/kevinfjiang/opendataval/tree/main/opendataval/dataloader/readme.md)** | Stable | [Docs](https://github.com/kevinfjiang/opendataval/releases) | Embeddings available for image/NLP datasets |
| **[Models](https://github.com/kevinfjiang/opendataval/tree/main/opendataval/model/readme.md)** | Stable | [Docs](https://opendataval.github.io/opendataval.model.html#module-opendataval.model) | Support available for sk-learn models |
| **[Data Evaluators](https://github.com/kevinfjiang/opendataval/tree/main/opendataval/dataval/readme.md)** | Stable | [Docs](https://opendataval.github.io/opendataval.dataval.html#module-opendataval.dataval) | |
| **[Experiments](https://github.com/kevinfjiang/opendataval/tree/main/opendataval/experiment/readme.md)** | Stable | [Docs](https://opendataval.github.io/opendataval.experiment.html#module-opendataval.experiment) | |
| **[CLI](https://github.com/kevinfjiang/opendataval/tree/main/opendataval/__main__.py)** | Experimental | `opendataval --help` | No support for null values |

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

## Installation
----
1. Clone the repo
   ```sh
   git clone https://github.com/kevinfjiang/opendataval.git
   ```
2. Install dependencies
   ```sh
   make install
   ```
    a. Install optional dependencies if you're contributing  TODO contributing.md
    ```sh
    make install-dev
    ```
    b. If you want to pull in kaggle datasets, I'd reccomend looking how to add a kaggle folder to the current directory. Tutorial [here](https://www.analyticsvidhya.com/blog/2021/04/how-to-download-kaggle-datasets-using-jupyter-notebook/)

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## :zap: Quick Start
----
To set up an experiment on DataEvaluators. Feel free to change the source code as needed for a project.

```python
from opendataval.experiment import ExperimentMediator

exper_med = ExperimentMediator.model_factory_setup(
    dataset_name='iris',
    force_download=False,
    train_count=100,
    valid_count=50,
    test_count=50,
    model_name='ClassifierMLP',
    train_kwargs={'epochs': 5, 'batch_size': 20},
)
list_of_data_evaluators = [ChildEvaluator(), ...]  # Define evaluators here
eval_med = exper_med.compute_data_values(list_of_data_evaluators)

# Runs a discover the noisy data experiment for each DataEvaluator and plots
data, fig = eval_med.plot(discover_corrupted_sample)

# Runs non-plottable experiment
data = eval_method.evaluate(noisy_detection)
```

### :computer: CLI
`opendataval` comes with a quick [CLI](https://github.com/kevinfjiang/opendataval/tree/main/opendataval/__main__.py) tool, The tool is under development and the template for a csv input is found at [`cli.csv`](https://github.com/kevinfjiang/opendataval/tree/main/cli.csv). Note that for kwarg arguments, the input must be valid json.

To use run the following command if installed with make-install:
```sh
opendataval --file cli.csv -n [job_id] -o [path/to/file/]
```
To run without installing the script:
```
python opendataval --file cli.csv -n [job_id] -o [path/to/file/]
```

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

### :control_knobs: API
Here are the 4 interacting parts of opendataval
1. `DataFetcher`, Loads data and holds meta data regarding splits
2. `Model`, trainable prediction model.
3. `DataEvaluator`, Measures the data values of input data point for a specified model.
4. `ExperimentMediator`, facilitates experiments regarding data values across several `DataEvaluator`s

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

#### [`DataFetcher`](https://github.com/kevinfjiang/opendataval/tree/main/opendataval/dataloader/readme.md)
The DataFetcher takes the name of a [`Register`](https://github.com/kevinfjiang/opendataval/tree/main/opendataval/dataloader/readme.md#register-datasets) dataset and loads, transforms, splits, and adds noise to the data set.
```python
from opendataval.dataloader import DataFetcher

DataFetcher.datasets_available()  # ['dataset_name1', 'dataset_name2']
fetcher = DataFetcher(dataset_name='dataset_name1')

fetcher = fetcher.split_dataset_by_count(70, 20, 10)
fetcher = fetcher.noisify(mix_labels, noise_rate=.1)

x_train, y_train, x_valid, y_valid, x_test, y_test = fetcher.datapoints
```

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

#### [`Model`](https://github.com/kevinfjiang/opendataval/tree/main/opendataval/model/readme.md)
`Model` is the predictive model for Data Evaluators.

```python
from opendataval.model import LogisticRegression

model = LogisticRegression(input_dim, output_dim)

model.fit(x, y)
model.predict(x)
>>> torch.Tensor(...)
```

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

#### [`DataEvaluator`](https://github.com/kevinfjiang/opendataval/tree/main/opendataval/dataval/readme.md)
We have a catalog of `DataEvaluator` to run experiments. To do so, input the `Model`, `DataFetcher`, and an evaluation metric (such as accuracy).

```python
from opendataval.dataval.ame import AME

dataval = (
    AME(num_models=8000)
    .train(fetcher=fetcher, pred_model=model, metric=metric)
)

data_values = dataval.data_values  # Cached values
data_values = dataval.evaluate_data_values()  # Recomputed values
>>> np.ndarray([.888, .132, ...])
```
<p align="right">(<a href="#readme-top">Back to top</a>)</p>

#### [`ExperimentMediator`](https://github.com/kevinfjiang/opendataval/tree/main/opendataval/experiment/readme.md)
`ExperimentMediator` is helps make a cohesive and controlled experiment. NOTE Warnings are raised if errors occur in a specific `DataEvaluator`.
```python
expermed = ExperimentrMediator(fetcher, model, train_kwargs, metric_name).compute_data_values(data_evaluators)
```

Run experiments by passing in an experiment function: `(DataEvaluator, DataFetcher, ...) - > dict[str, Any]`. There are 5 found `exper_methods.py` with three being plotable.
```python
df = expermed.evaluate(noisy_detection)
df, figure = expermed.plot(discover_corrupted_sample)
```

For more examples, please refer to the [Documentation](opendataval.github.io)

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

## :medal_sports: opendataval Leaderboards
For datasets that start with the prefix challenge, we provide [leaderboards](https://opendataval.github.io/leaderboards). Compute the data values with an `ExperimentMediator` and use the `save_dataval` function to save a csv. Upload it to [here](https://opendataval.github.io/upload)!

```python
exper_med = ExperimentMediator.model_factory_setup(
    dataset_name='challenge-...', model_name=model_name, train_kwargs={...}, metric_name=metric_name
)
exper_med.compute_data_values([custom_data_evaluator]).evaluate(save_dataval, save_output=True)
```

<p align="right">(<a href="#readme-top">Back to top</a>)</p>



<!-- CONTRIBUTING -->
## :wave: Contributing

If you have a quick suggestion, reccomendation, bug-fixes please open an [issue]([issues-url]).
If you want to contribute to the project, either through data sets, experiments, presets, or fix stuff, please see our [Contribution page](CONTRIBUTING.md).

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

## :bulb: Vision
* **clean, descriptive specification syntax** -- based on modern object-oriented design principles for data science.
* **fair model assessment and benchmarking** -- Easily build and evaluate your Data Evaluators
* **easily extensible** -- Easily add your own data sets,

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

## :books: Cite us
TODO

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<!-- LICENSE -->
## :classical_building: License

Distributed under the MIT License. See [`LICENSE.txt`][license-url] for more information.

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/kevinfjiang/opendataval.svg?style=for-the-badge
[contributors-url]: https://github.com/kevinfjiang/opendataval/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/kevinfjiang/opendataval.svg?style=for-the-badge
[forks-url]: https://github.com/kevinfjiang/opendataval/network/members
[stars-shield]: https://img.shields.io/github/stars/kevinfjiang/opendataval.svg?style=for-the-badge
[stars-url]: https://github.com/kevinfjiang/opendataval/stargazers
[issues-shield]: https://img.shields.io/github/issues/kevinfjiang/opendataval.svg?style=for-the-badge
[issues-url]: https://github.com/kevinfjiang/opendataval/issues
[license-shield]: https://img.shields.io/github/license/kevinfjiang/opendataval.svg?style=for-the-badge
[license-url]: https://github.com/kevinfjiang/opendataval/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[test-url]: https://img.shields.io/github/actions/workflow/status/kevinfjiang/opendataval/test?style=for-the-badge
[test-shield]: https://img.shields.io/github/actions/workflow/status/kevinfjiang/opendataval/test?style=for-the-badge
[PyTorch-shield]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[scikit-learn-shield]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/stable/
[numpy-url]: https://numpy.org/
[numpy-shield]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white\
[release-shield]: https://img.shields.io/github/v/release/kevinfjiang/opendataval?style=for-the-badge
[release-url]: https://github.com/kevinfjiang/opendataval/releases
<!-- Pytest Coverage Comment:Start -->
[coverage_badge]: https://img.shields.io/badge/Coverage-85%25-green.svg?style=for-the-badge
<!-- Pytest Coverage Comment:End -->
