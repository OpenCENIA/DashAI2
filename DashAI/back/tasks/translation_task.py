from datasets import Dataset
from datasets.dataset_dict import DatasetDict

from DashAI.back.tasks.base_task import BaseTask


class TranslationTask(BaseTask):
    """
    Abstract class for translation tasks.
    Here you can change the methods provided by class TranslationTask.
    """

    name: str = "TranslationTask"
    source: str = ""
    target: str = ""

    def parse_input(self, input_data):
        d = {
            "train": Dataset.from_dict(
                {"x": input_data["train"]["x"], "y": input_data["train"]["y"]},
            ),
            "test": Dataset.from_dict(
                {"x": input_data["test"]["x"], "y": input_data["test"]["y"]},
            ),
        }

        d = DatasetDict(d)
        return d
