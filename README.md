<!-- Improved compatibility of Back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/kevinfjiang/opendataval">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Open Data Valuation</h3>

  <p align="center">
    Data Valuation Meta-Meta Framework
    <br />
    <a href="https://github.com/kevinfjiang/opendataval"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/kevinfjiang/opendataval">View Demo</a>
    ·
    <a href="https://github.com/kevinfjiang/opendataval/issues">Report Bug</a>
    ·
    <a href="https://github.com/kevinfjiang/opendataval/issues">Request Feature</a>
  </p>
</div>

# DataOob : Towards a Transparent Data Valuations

----

[**Website**](TODO) | [**arXiv Paper**](TODO)
<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![Build][test-shield]][test-url]
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue?style=for-the-badge)](https://www.python.org/)
<!-- Pytest Coverage Comment:Begin -->
![Coverage](https://img.shields.io/badge/Coverage-87%25-green.svg?style=for-the-badge)
<!-- Pytest Coverage Comment:End -->


**dataoob** is the first general-purpose lightweight library that provides a comprehensive list of functions to systematically evaluate the data valuations of several data valuation meta-frameworks. dataoob supports the development of new datasets (both synthetic and real-world) and explanation methods, with a strong bent towards promoting systematic, reproducible, and transparent evaluation of data values. TODO most of this is filler from OpenXAI, to change

**dataoob** is an open-source initiative that comprises of a collection of curated datasets, models, and data value estimators, and provides a simple and easy-to-use API that enables researchers and practitioners to benchmark explanation methods using just a few lines of code.



<!-- TABLE OF CONTENTS -->
<details>
  <summary>👇Table of Contents</summary>
  <hr>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#usage">Usage</a></li>
      </ul>
    </li>
    <li><a href="#architecture">Architecture</a></li><li>
    <a href="#opendataval-leaderboards">Leaderboards</a></li>
    <li><a href="#contributing">Contributing</a></li><ul>
            <li><a href="#roadmap">Roadmap</a></li>
    </ul></li>
    <li><a href="#license">License</a></li>
    <li><a href="#cite-us">Cite Us</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

<!-- Here's a blank template to get started: To avoid retyping too much info. Do a search and replace with your text editor for the following: `kevinfjiang`, `opendataval`, `twitter_handle`, `yahoo`, `kevinfjiang`, `Open Data Val`, `Data Valuation Meta-Meta Framework` -->

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

#### Built with

* [![Pytorch][PyTorch-shield]][PyTorch-url]
* [![scikit-learn][scikit-learn-shield]][scikit-learn-url]
* [![numpy][numpy-shield]][numpy-url]
* [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge&logo=appveyor)](https://github.com/psf/black)



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

Install Python3.10 or Python 3.11.
* Brew
  ```sh
  brew install python@3.11
  python3.11 --version
  ```
* Linux
  ```sh
  sudo apt-get install python3.11
  python3.11 --version
  ```
* Anaconda (reccomended as creates an env)
  ```sh
  conda create --name [env_name] -c conda-forge python=3.10
  conda activate [env_name]
  python --version
  ```

### Installation

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
3. Open demo.ipynb to see presets and the training workflow

<p align="right">(<a href="#readme-top">Back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Quick Start
To set up an experiment on DataEvaluators
```python
from dataoob.evaluator import preset
from dataoob.eval_method import discover_corrupted_sample, noisy

eval_med = preset.from_presets('iris_low_noise_ann', 'experiment')  # Allocate 20+ min

# Runs a discover the noisy data experiment for each DataFetcher
# Plots the results in figure (if has plot arg), and returns the results as a DataFrame
data, fig = eval_med.plot(discover_corrupted_sample)

# If not plot: Axes argument, we can still evaluate the function with the following
# Any function with arguments (DataEvaluator, DataFetcher, ...) -> dict is valid.
data = eval_method.evaluate(noisy_detection)
```

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

### Overview
Here are the 4 interacting parts of opendataval
1. `DataEvaluator`, Measures the data values of input data point for a specified model.
2. `DataFetcher`, Loads data and holds meta data regarding splits
3. `Model`, trainable prediction model.
4. `ExperimentMediator`, facilitates experiments regarding data values across several `DataEvaluator`s
```python
fetcher = (
    DataFetcher(dataset_name='name')
    .split_dataset(train_count=.7, valid_count=.2, test_count=.1)
    .noisify(noise_func)  # (DataFetcher, ...) -> dict
)

evaluator = ChildEvaluator()  # DataEvaluator is abstract
model = ChildModel()  # Model is abstract

expermed = ExperimentMediator(
    fetcher=fetcher,
    data_evaluators=[evaluator],
    pred_model=model,
    metric_name='accuracy'
)

expermed.evaluate(exper_func)
```

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

### `Model`
`Model` is an abstract base class that requires the implementation of three methods: `.train(x, y)`, `.predict(x)`, `.clone()`.

`Model` was primarily designed with PyTorch models in mind, which is why there are additional mixins to inherit for PyTorch models. However, there is support for sk-learn models through a wrapper:
```python
from dataoob.model import ClassifierSkLearnWrapper
from sklearn.linear_model import LogisticRegression

wrapped_lr = ClassifierSkLearnWrapper(LogisticRegression(), label_dim, device=torch.device('...'))

wrapped_lr.fit(x, y)
wrapped_lr.predict(x)  # equiv of `.predict_proba()`
```
<p align="right">(<a href="#readme-top">Back to top</a>)</p>

### `DataEvaluator`
We have a catalog of `DataEvaluator` with many default arguments. Many DataEvaluators will do well with default arguments for data points `n`<100. To pull one in do the following
```python
from dataoob.dataval.ame import AME

dataval = AME()
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
<p align="right">(<a href="#readme-top">Back to top</a>)</p>
### `DataFetcher`
The DataFetcher accepts the name of a `Register` data set and handles the preprocessing involved. For our purposes, we can find the registered datasets with:
```python
DataFetcher.datasets_available()  # ['name1', 'name2']
```

A fetcher first takes a data set name to be fetched.
```python
from dataoob.dataloader import DataFetcher

fetcher = DataFetcher(dataset_name='name1')
```

From there we must define how we will split the data set into train/valid/test splits
```python
fetcher = fetcher.split_dataset(70, 20, 10)  # Data set counts
fetcher = fetcher.split_dataset(.7, .2, .1)  # Replits on proportions
```

To get data points
```python
x_train, y_train, x_valid, y_valid, x_test, y_test = fetcher.datapoints
```

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

### `ExperimentMediator`
`ExperimentMediator` is helps make a cohesive and controlled experiment. By injecting a model, data loader, and dataevaluators, it will train the models and faciliatate additional experiments. NOTE when training `DataEvaluator`, errors might be raised and caught. A warning will be raised but training will continue because training can often take a long time.
```python
expermed = ExperimentrMediator(
    loader, data_evaluators, model, train_kwargs, metric_name
)
```

From here we can run experiments by passing in an experiment function `(DataEvaluator, DataFetcher) - > dict[str, Any]`. There are 5 found `exper_methods.py` with three being plotable. All returns include a pandas `DataFrame`.
```python
df = expermed.evaluate(noisy_detection)
df, figure = expermed.plot(discover_corrupted_sample)
```

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<!-- UML Map -->
## Architecture
[![](https://mermaid.ink/img/pako:eNqdVdtu2zAM_RVBe3FaJ7Pr5mYUBbZentaXrtvDYMBQbCUV4EiGLXfxiuTbR0mJrThxByxAEok8pMhDinrHiUgpDnGSkbK8Z2RVkDVSn4ir32e6YqWkBRoOL9A9keSbIClsQ3Qn1rkomWSCf34kCUiNhXbU2L0boVaIN1IwImnZyjKyoJm132XgPk7hHGdgpNuImwVJ05gLVtJ4WfEE3QyHt1ZAIfqRp8o3UsboN5OvSKPtoI5dWKHd3CiBSuX2FrVix0o4038DSDxliTzQ00Rnfi18hP0IA2tCrUajC9go7cMbySoihWLwy2pV0BVRxxrzJyhF9l-WJj_reCu5poSKmZLKVsNTUhSkRptYFoTxU0Xdp9jEEA1Lz1n0KBTvdcw4sGc3wGWZZ0zG-9AcfVyciIpLF2lPZjOwLJQntqydTkMsuw3TctJQZ7FiuM4LmsZrtWw1L1We0d0L5aUodpq0XDAu7aAZzytp7BzjSK8HJxDdyaddZOFMxgqnmKto6dhaaiKnxwCkKT2bro7GSvNyyaRjckEbF-1XtX2G4gBaukEN9qiO_77ehtXDJqcFW1Mun8CVIro1adu26eZ_G3YvQi9cXTw7_VOExcVJHVrVN7ghu6NwTeVjetiXH7XOmYo5VIVyaE199iOMVmoTnwnZD-vQfyZ3PQJb-xA9V7w0Ag1Egh-XwGaqteuZg50x2NaxocS1GHU7A7KJHiHs4jUt1oSl8MzowyIsXynkiENYLuDiR9i15D_VM7HIYEwAYB9dhHNIihT1ncggEW35KV3QgC72xhbmhW6kjQu8YBwkp7ivooCQP0RqspS_PkBJE6Fm3FFgnjfxPc9CwfiVrAMaJ7PrxVlX_XEpKJC6BU5JJcX3mic4lEVFXVzp92__gh-E0CpQqKf9Ey_4kq3ANCf8lxANCLY4fMcbHF75s5HvT725N_bn1-PZ3MU1DoP5yPcmQRCMp17gj6-mWxf_0fbeaHblT64n09nMh-_2L3G-nKE?type=png)](https://mermaid.live/edit#pako:eNqdVdtu2zAM_RVBe3FaJ7Pr5mYUBbZentaXrtvDYMBQbCUV4EiGLXfxiuTbR0mJrThxByxAEok8pMhDinrHiUgpDnGSkbK8Z2RVkDVSn4ir32e6YqWkBRoOL9A9keSbIClsQ3Qn1rkomWSCf34kCUiNhXbU2L0boVaIN1IwImnZyjKyoJm132XgPk7hHGdgpNuImwVJ05gLVtJ4WfEE3QyHt1ZAIfqRp8o3UsboN5OvSKPtoI5dWKHd3CiBSuX2FrVix0o4038DSDxliTzQ00Rnfi18hP0IA2tCrUajC9go7cMbySoihWLwy2pV0BVRxxrzJyhF9l-WJj_reCu5poSKmZLKVsNTUhSkRptYFoTxU0Xdp9jEEA1Lz1n0KBTvdcw4sGc3wGWZZ0zG-9AcfVyciIpLF2lPZjOwLJQntqydTkMsuw3TctJQZ7FiuM4LmsZrtWw1L1We0d0L5aUodpq0XDAu7aAZzytp7BzjSK8HJxDdyaddZOFMxgqnmKto6dhaaiKnxwCkKT2bro7GSvNyyaRjckEbF-1XtX2G4gBaukEN9qiO_77ehtXDJqcFW1Mun8CVIro1adu26eZ_G3YvQi9cXTw7_VOExcVJHVrVN7ghu6NwTeVjetiXH7XOmYo5VIVyaE199iOMVmoTnwnZD-vQfyZ3PQJb-xA9V7w0Ag1Egh-XwGaqteuZg50x2NaxocS1GHU7A7KJHiHs4jUt1oSl8MzowyIsXynkiENYLuDiR9i15D_VM7HIYEwAYB9dhHNIihT1ncggEW35KV3QgC72xhbmhW6kjQu8YBwkp7ivooCQP0RqspS_PkBJE6Fm3FFgnjfxPc9CwfiVrAMaJ7PrxVlX_XEpKJC6BU5JJcX3mic4lEVFXVzp92__gh-E0CpQqKf9Ey_4kq3ANCf8lxANCLY4fMcbHF75s5HvT725N_bn1-PZ3MU1DoP5yPcmQRCMp17gj6-mWxf_0fbeaHblT64n09nMh-_2L3G-nKE)

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

## Opendataval Leaderboards
For every `DataEvaluator`, we will provide its performance compared to the other
`DataEvaluators` for a specific preset specified in `presets`.

Submitting a new `DataEvaluator` and a list of `presets` and `experiments` to test it on
will be trained and tested several times. We will always use the same seeds, with one
seed used public: `42`.

Submitting a new `Register` data set is very helpful as that will increases the number
of different presets.

Submitting a new `exper_func` increases the number of metrics for evaluation.

<p align="right">(<a href="#readme-top">Back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

If you have a quick suggestion, reccomendation, bug-fixes please open an [issue](https://github.com/kevinfjiang/opendataval/issues).
If you want to contribute to the project, either through data sets, experiments, presets, or fix stuff, please see our [Contribution page](CONTRIBUTING.md).
If you have created a new `DataEvaluator`, and would like to get it implemented and
evaluated, please see our [Contribution page](CONTRIBUTING.md). TODO i haven't figured out how to evaluate but we'll leave it like this for now.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/kevinfjiang/opendataval/issues) for a full list of proposed features (and known issues).
## Cite us
TODO

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">Back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - kevinfjiang@yahoo.com

<p align="right">(<a href="#readme-top">Back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

## Sharing TODO delete me
To share this repository with non-contributers prior to making this public, run the following command.
Expiration date:
![Token expiration](https://img.shields.io/date/1683432000?style=for-the-badge)
```sh
git clone https://oauth2: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@github.com/kevinfjiang/opendataval.git
```

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
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
[PyTorch-shield]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[scikit-learn-shield]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/stable/
[numpy-url]: https://numpy.org/
[numpy-shield]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white