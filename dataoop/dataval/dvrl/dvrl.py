from sklearn import metrics

from dataoop.dataval import Evaluator, Model

from .dvrlgoog import Dvrl


class DVRL(Evaluator):
    def __init__(
        self, model: Model, parameters: dict[str, str], flags: dict[str, bool]
    ):
        super().__init__(model)
        self.problem = "classification"
        self.metric = (
            metrics.mean_squared_error
        )  # Find away to infer these two from the parent evaluator,
        # or an option to change this

        self.parameters = parameters
        self.flags = flags

    def data_value_evaluator(self, x_train, y_train, x_valid, y_valid):
        dvrl = Dvrl(
            x_train,
            y_train,
            x_valid,
            y_valid,
            problem=self.problem,
            pred_model=self.model,
            parameters=self.parameters,
            checkpoint_file_name="cp",
            flags=self.flags,
        )
        dvrl.train_dvrl(self.metric)
        dve_out = dvrl.data_valuator(x_train, y_train)
        return dve_out
