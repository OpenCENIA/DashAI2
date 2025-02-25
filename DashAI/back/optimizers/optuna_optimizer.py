import optuna

from DashAI.back.core.schema_fields import (
    BaseSchema,
    enum_field,
    int_field,
    schema_field,
)
from DashAI.back.optimizers.base_optimizer import BaseOptimizer


class OptunaSchema(BaseSchema):
    n_trials: schema_field(
        int_field(gt=0),
        placeholder=10,
        description="The parameter 'n_trials' is the quantity of trials"
        "per study. It must be of type positive integer.",
    )  # type: ignore
    sampler: schema_field(
        enum_field(
            enum=[
                "TPESampler",
                "CmaEsSampler",
                "GridSampler",
                "GPSampler",
                "NSGAIISampler",
                "QMCSampler",
                "RandomSampler",
            ]
        ),
        placeholder="TPESampler",
        description="Coefficient for 'rbf', 'poly' and 'sigmoid' kernels"
        ". Must be in string format and can be 'scale' or 'auto'.",
    )  # type: ignore
    pruner: schema_field(
        enum_field(enum=["MedianPruner", "None"]),
        placeholder="None",
        description="Coefficient for 'rbf', 'poly' and 'sigmoid' kernels"
        ". Must be in string format and can be 'scale' or 'auto'.",
    )  # type: ignore


class OptunaOptimizer(BaseOptimizer):
    SCHEMA = OptunaSchema

    COMPATIBLE_COMPONENTS = [
        "TabularClassificationTask",
        "TextClassificationTask",
        "TranslationTask",
        "RegressionTask",
    ]

    def __init__(self, n_trials=None, sampler=None, pruner=None):
        self.n_trials = n_trials
        self.sampler = getattr(optuna.samplers, sampler)
        self.pruner = pruner

    def optimize(self, model, input_dataset, output_dataset, parameters, metric, task):
        """
        Optimization process

        Args:
            model (class): class for the model from the current experiment
            dataset (dict): dict with the data to train and validation
            parameters (dict): dict with the information to create the search space
            metric (class): class for the metric to optimize

        Returns
        -------
            None
        """
        self.model = model
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset
        self.parameters = parameters

        if metric["name"] in ["Accuracy", "F1", "Precision", "Recall"]:
            study = optuna.create_study(
                direction="maximize", sampler=self.sampler(), pruner=self.pruner
            )
        else:
            study = optuna.create_study(
                direction="minimize", sampler=self.sampler(), pruner=self.pruner
            )

        self.metric = metric["class"]

        if task == "TextClassificationTask":

            def objective(trial):
                classifier_trial = self.model.classifier
                for hyperparameter, values in self.parameters.items():
                    value = trial.suggest_int(hyperparameter, values[0], values[-1])
                    setattr(classifier_trial, hyperparameter, value)

                model_trial = self.model
                model_trial.classifier = classifier_trial
                model_trial.fit(
                    self.input_dataset["train"], self.output_dataset["train"]
                )
                y_pred = model_trial.predict(input_dataset["validation"])
                score = self.metric.score(output_dataset["validation"], y_pred)

                return score

        else:

            def objective(trial):
                model_trial = self.model
                for hyperparameter, values in self.parameters.items():
                    value = trial.suggest_int(hyperparameter, values[0], values[-1])
                    setattr(model_trial, hyperparameter, value)

                model_trial.fit(
                    self.input_dataset["train"], self.output_dataset["train"]
                )
                y_pred = model_trial.predict(input_dataset["validation"])
                score = self.metric.score(output_dataset["validation"], y_pred)

                return score

        study.optimize(objective, n_trials=self.n_trials)

        best_params = study.best_params
        best_model = self.model
        for hyperparameter, value in best_params.items():
            setattr(best_model, hyperparameter, value)
        best_model.fit(self.input_dataset["train"], self.output_dataset["train"])
        self.model = best_model
        self.study = study

    def get_model(self):
        return self.model

    def get_trials_values(self):
        trials = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trials.append({"params": trial.params, "value": trial.value})
        return trials
