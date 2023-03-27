import sklearn.metrics as metrics
import torch

from dataoob.dataloader.data_loader import DataLoader
from dataoob.dataval import DataEvaluator

metrics_dict = {
    "accuracy": lambda a, b: metrics.accuracy_score(
        a.detach().cpu()[:, 1], torch.argmax(b.detach().cpu(), axis=1)
    )
}  # Maybe have some nice dataclasses to organize data loading kwargs, training kwargs
# Like have training kwarg, when you define the model
# an devaluating kwargs, evaluator kwargs
class EvaluatorPipeline:
    def __init__(
        self,
        dataset: str,
        noisy_rate: float,
        pred_model,
        metric: str,
        data_evaluators: list[DataEvaluator],
        device: torch.device = torch.device("cpu"),
        batch_size=32,
        epochs=10,
    ):
        force_redownload = False
        (
            (x_train, y_train),
            (x_valid, y_valid),
            (xt, yt),
            self.noisy_indices,
        ) = DataLoader(dataset, force_redownload, 100, 50, 0, noisy_rate, device)

        self.metric = metric
        self.data_evaluators = data_evaluators

        for data_val in self.data_evaluators:
            # TODO this could be a blast zone, wrap this in something
            data_val.input_model_metric(pred_model, metrics_dict[metric])
            data_val.train(x_train, y_train, x_valid, y_valid)

    def evaluate(self, evaluator, **eval_kwargs):
        res = []
        for data_val in self.data_evaluators:
            res.append(evaluator(data_val, noisy_index=self.noisy_indices))
        return res

    def plot(self):
        pass
