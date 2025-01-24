import json
import os
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from DashAI.back.dataloaders.classes.json_dataloader import JSONDataLoader
from DashAI.back.dependencies.database.models import Dataset, Experiment, Run
from DashAI.back.dependencies.registry import ComponentRegistry
from DashAI.back.job.model_job import ModelJob
from DashAI.back.metrics import BaseMetric
from DashAI.back.models import BaseModel
from DashAI.back.optimizers import OptunaOptimizer
from DashAI.back.tasks import BaseTask


@pytest.fixture(scope="module", name="dataset", autouse=True)
def create_dataset(client: TestClient):
    script_dir = os.path.dirname(__file__)
    test_dataset = "ImdbSentimentDatasetSmall.json"
    abs_file_path = os.path.join(script_dir, test_dataset)
    with open(abs_file_path, "rb") as json_file:
        response = client.post(
            "/api/v1/dataset/",
            data={
                "params": """{  "dataloader": "JSONDataLoader",
                                    "name": "test_json",
                                    "splits_in_folders": false,
                                    "splits": {
                                        "train_size": 0.5,
                                        "test_size": 0.2,
                                        "val_size": 0.3
                                    },
                                    "data_key": "data",
                                    "more_options": {
                                        "seed": 42,
                                        "shuffle": true,
                                        "stratify": true
                                    }
                                }""",
                "url": "",
            },
            files={"file": ("filename", json_file, "text/json")},
        )
    assert response.status_code == 201, response.text
    dataset = response.json()

    yield dataset

    response = client.delete(f"/api/v1/dataset/{dataset['id']}")
    assert response.status_code == 204, response.text


@pytest.fixture(scope="module", name="experiment_id", autouse=True)
def create_experiment(client: TestClient, dataset: Dataset):
    session_factory = client.app.container["session_factory"]

    with session_factory() as db:
        experiment = Experiment(
            dataset_id=dataset["id"],
            name="Experiment",
            task_name="TextClassificationTask",
            input_columns=["text"],
            output_columns=["class"],
            splits=json.dumps(
                {
                    "train": 0.5,
                    "test": 0.2,
                    "validation": 0.3,
                    "is_random": True,
                    "has_changed": True,
                    "seed": 42,
                    "shuffle": True,
                    "stratify": True,
                }
            ),
        )
        db.add(experiment)
        db.commit()
        db.refresh(experiment)

        yield experiment.id

        db.delete(experiment)
        db.commit()
        db.close()


@pytest.fixture(scope="module", name="run_id")
def create_run_id(client: TestClient, experiment_id: int):
    container = client.app.container
    session_factory = container["session_factory"]

    with session_factory() as db:
        run = Run(
            experiment_id=experiment_id,
            optimizer_name="OptunaOptimizer",
            optimizer_parameters={
                "n_trials": 10,
                "sampler": "TPESampler",
                "pruner": "None",
                "metric": "auto",
            },
            model_name="DistilBertTransformer",
            parameters={
                "num_train_epochs": 5,
                "batch_size": 8,
                "learning_rate": 5e-5,
                "device": "gpu",
                "weight_decay": 0.0,
            },
            name="Run",
        )
        db.add(run)
        db.commit()
        db.refresh(run)

        yield run.id

        db.delete(run)
        db.commit()
        db.close()


def test_create_trained_run(client: TestClient, run_id: int):
    response = client.post(
        "/api/v1/job/",
        json={"job_type": "ModelJob", "kwargs": {"run_id": run_id}},
    )
    assert response.status_code == 201, response.text

    response = client.post("/api/v1/job/start/?stop_when_queue_empties=True")
    assert response.status_code == 202, response.text

    return run_id
