## `Examples`
We offer a comprehensive collection of user-friendly notebook examples that greatly simplify the utilization of data values in practical data analysis. Our notebooks cover a wide range of data domains, including image and text analysis as well as various models (logistic regression, BERT, ResNet, Random Forest, Gradient Boosting, ...). Below, you'll find a summary of the current examples provided by `opendataval`:

| Task | Datasets | Prediction Model | Notes |
|---------|--------|-------|-------|
| classification | adult | LogisticRegression | Tabular, label noise |
| classification | nomao | LogisticRegression | Tabular, label noise |
| classification | synthetic | RandomForest | Custom Tabular Dataset, label noise |
| classification | pol | Gradient Boosting | Tabular Dataset, label noise, custom model paramters |
| classification | BBC-embeddings | DistilBertModel + LogisticRegression | Text, feature noise |
| classification | IMDB-embeddings | DistilBertModel + LogisticRegression | Text, label noise |
| classification | CIFAR-embeddings | ResNet50 + LogisticRegression | Images, label noise |
| classification | BBC | DistilBertModel | Text (very slow; not recommended) |
| regression | diabetes | LinearRegression | Tabular, feature noise |
| regression | wave_energy | Gradient Boosting | Tabular, feature noise |
| regression | stock | Multi-Layer Perceptron | Tabular, feature noise |

Are you interested in sharing your data analysis? Please see our [Contribution page](https://github.com/opendataval/opendataval/blob/main/CONTRIBUTING.md).