import json
from abc import ABCMeta

from Models.enums.squema_types import SquemaTypes

dict_squemas = {
    SquemaTypes.model: "Models/parameters/models_schemas/",
    SquemaTypes.preprocess: "Models/parameters/preprocess_schemas",
    SquemaTypes.dataloader: "Dataloaders/params_schemas/",
    SquemaTypes.task: "TaskLib/tasks_schemas/"
}


class ConfigObject(metaclass=ABCMeta):
    @staticmethod
    def get_squema(type, name):
        try:
            print(f"{dict_squemas[type]}{name}.json")
            f = open(f"{dict_squemas[type]}{name}.json")
        except FileNotFoundError:
            f = open(f"{dict_squemas[type]}{name.lower()}.json")
        return json.load(f)
