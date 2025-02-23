import os
import shutil

import numpy as np
import pytest
import torch

from DashAI.back.dataloaders.classes.dashai_dataset import (
    select_columns,
    split_dataset,
    split_indexes,
    to_dashai_dataset,
)
from DashAI.back.dataloaders.classes.json_dataloader import JSONDataLoader
from DashAI.back.models.hugging_face.distilbert_transformer import DistilBertTransformer


@pytest.fixture(scope="module", name="splited_dataset")
def splited_dataset_fixture():
    test_dataset_path = "tests/back/models/dummy_text.json"
    dataloader_test = JSONDataLoader()

    datasetdict = dataloader_test.load_data(
        filepath_or_buffer=test_dataset_path,
        temp_path="tests/back/models",
        params={"data_key": "data"},
    )

    datasetdict = to_dashai_dataset(datasetdict)

    train_idx, test_idx, val_idx = split_indexes(
        total_rows=len(datasetdict),
        train_size=0.6,
        test_size=0.2,
        val_size=0.2,
    )

    splited_dataset = split_dataset(
        datasetdict,
        train_indexes=train_idx,
        test_indexes=test_idx,
        val_indexes=val_idx,
    )

    x, y = select_columns(
        splited_dataset,
        ["text"],
        ["class"],
    )
    x = split_dataset(x)
    y = split_dataset(y)
    y["train"] = y["train"].map(lambda example: {"class": int(example["class"])})
    y["test"] = y["test"].map(lambda example: {"class": int(example["class"])})
    y["validation"] = y["validation"].map(
        lambda example: {"class": int(example["class"])}
    )

    return (x, y)


@pytest.fixture()
def sample_model():
    model = DistilBertTransformer(
        num_train_epochs=2,
        batch_size=16,
        learning_rate=5e-5,
        device="cpu",
        weight_decay=0.01,
        num_labels=2,
    )
    return model


def test_model_initialization(sample_model):
    assert sample_model.model is not None
    assert sample_model.tokenizer is not None
    assert sample_model.model_name == "distilbert-base-uncased"
    assert sample_model.fitted is False


# Prueba de Tokenización
def test_tokenize_data(sample_model, splited_dataset):
    x, y = splited_dataset
    x = x["train"]
    tokenized_dataset = sample_model.tokenize_data(x)

    assert "input_ids" in tokenized_dataset.features
    assert "attention_mask" in tokenized_dataset.features
    assert len(tokenized_dataset) == len(x)


def test_fit(sample_model, splited_dataset):
    x_train, y_train = splited_dataset
    x_train = x_train["train"]
    y_train = y_train["train"]
    assert all(isinstance(label, int) for label in y_train["class"])
    sample_model.fit(x_train, y_train)
    assert sample_model.fitted is True


def test_predict(sample_model, splited_dataset):
    x_train, y_train = splited_dataset
    x_train = x_train["train"]
    y_train = y_train["train"]

    sample_model.fit(x_train, y_train)

    predictions = sample_model.predict(x_train)

    assert isinstance(predictions, list)
    assert len(predictions) == len(x_train)
    assert all(isinstance(pred, np.ndarray) for pred in predictions)
    assert all(pred.shape == (2,) for pred in predictions)
    assert all(np.isclose(np.sum(pred), 1.0) for pred in predictions)


def test_save_and_load(sample_model, splited_dataset, tmp_path):
    x_train, y_train = splited_dataset
    x_train = x_train["train"]
    y_train = y_train["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_model.fit(x_train, y_train)
    save_path = os.path.join(tmp_path, "distilbert_model")

    sample_model.save(save_path)

    loaded_model = sample_model.load(save_path)

    assert loaded_model.fitted, "Model is not fitted after loading"

    sample_model.model.to(device)
    loaded_model.model.to(device)

    original_state_dict = sample_model.model.state_dict()
    loaded_state_dict = loaded_model.model.state_dict()

    assert original_state_dict.keys() == loaded_state_dict.keys()

    for key in original_state_dict:
        assert torch.equal(
            original_state_dict[key], loaded_state_dict[key]
        ), f"""The loaded model should have the same weights and parameters
        as the original model (mismatch in {key})"""

    shutil.rmtree(save_path)
