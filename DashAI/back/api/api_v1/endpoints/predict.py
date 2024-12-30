import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, status
from fastapi.exceptions import HTTPException
from kink import di, inject
from sqlalchemy.orm import sessionmaker

from DashAI.back.api.api_v1.schemas.predict_params import (
    RenameRequest,
    filterDatasetParams,
)
from DashAI.back.dataloaders.classes.dashai_dataset import get_columns_spec
from DashAI.back.dependencies.database.models import Dataset, Experiment, Run

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/metadata_json/")
@inject
async def get_metadata_prediction_json(config: dict = Depends(lambda: di["config"])):
    """
    Fetches prediction metadata from JSON files.

    Parameters
    ----------
    config : dict
        Configuration dictionary injected automatically.

    Returns
    -------
    List[dict]
        A list of metadata dictionaries from prediction JSON files.

    Raises
    ------
    HTTPException
        If the directory or files cannot be accessed.
    """
    path = Path(f"{config['DATASETS_PATH']}/predictions/")
    try:
        path.mkdir(parents=True, exist_ok=True)
        files = os.listdir(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    json_files = [f for f in files if f.endswith(".json")]
    if not json_files:
        return []

    prediction_data = []
    # Read and collect metadata from each JSON file
    for json_file in json_files:
        file_path = path / json_file
        with open(file_path, "r") as f:
            data = json.load(f)["metadata"]
            prediction_data.append(data)
    return prediction_data


@router.get("/prediciton_table")
@inject
async def get_prediction_table(
    session_factory: sessionmaker = Depends(lambda: di["session_factory"]),
):
    """
    Fetches a table of prediction metadata from the database.

    Parameters
    ----------
    session_factory : sessionmaker
        SQLAlchemy session factory injected automatically.

    Returns
    -------
    List[dict]
        A list of dictionaries containing prediction metadata.

    Raises
    ------
    HTTPException
        If no data is found.
    """

    with session_factory() as db:
        query_results = db.query(
            Experiment.task_name,
            Run.model_name.label("run_type"),
            Dataset.name.label("dataset_name"),
            Dataset.id.label("dataset_id"),
            Dataset.model_name.label("dataset_model_name"),
            Dataset.last_modified,
        ).all()

        if not query_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No data found",
            )

        prediction_data = [
            {
                "id": result.dataset_id,
                "last_modified": result.last_modified,
                "run_name": result.run_type,
                "model_name": result.dataset_model_name,
                "dataset_name": result.dataset_name,
                "task_name": result.task_name,
            }
            for result in query_results
        ]
        return prediction_data


@router.get("/model_table")
@inject
async def get_model_table(
    session_factory: sessionmaker = Depends(lambda: di["session_factory"]),
):
    """
    Fetches a table of model metadata from the database.

    Parameters
    ----------
    session_factory : sessionmaker
        SQLAlchemy session factory injected automatically.

    Returns
    -------
    List[dict]
        A list of dictionaries containing model metadata.

    Raises
    ------
    HTTPException
        If no data is found.
    """
    with session_factory() as db:
        query_results = (
            db.query(
                Experiment.id.label("experiment_id"),
                Experiment.name.label("experiment_name"),
                Experiment.created,
                Experiment.task_name,
                Run.name.label("run_name"),
                Run.model_name,
                Dataset.name.label("dataset_name"),
                Dataset.id.label("dataset_id"),
            )
            .join(Experiment, Experiment.id == Run.experiment_id)
            .join(Dataset, Experiment.dataset_id == Dataset.id)
            .all()
        )
        if not query_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No data found",
            )

        prediction_data = [
            {
                "id": result.experiment_id,
                "experiment_name": result.experiment_name,
                "created": result.created,
                "run_name": result.run_name,
                "task_name": result.task_name,
                "model_name": result.model_name,
                "dataset_name": result.dataset_name,
                "dataset_id": result.dataset_id,
            }
            for result in query_results
        ]
        return prediction_data


@router.post("/filter_datasets")
async def filter_datasets_endpoint(
    params: filterDatasetParams,
    session_factory: sessionmaker = Depends(lambda: di["session_factory"]),
):
    """
    Filter datasets that match the column specifications of the train dataset.

    Parameters
    ----------
    train_dataset_id : int
        The ID of the train dataset.
    datasets : List[str]
        List of datasets paths to filter.

    Returns
    -------
    List[Dataset]
        List of datasets that match the column specifications of the train dataset.
    """
    try:
        with session_factory() as db:
            train_dataset_id = params.train_dataset_id
            datasets_paths = params.datasets
            filtered_list = []
            file_path = f"{db.get(Dataset, train_dataset_id).file_path}\\dataset"
            if not file_path:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Dataset not found",
                )
            train_dataset_spec = get_columns_spec(file_path)
            for dataset_path in datasets_paths:
                # dataset_path = f"{dataset_path}\\dataset"
                dataset_spec = get_columns_spec(f"{dataset_path}\\dataset")
                if train_dataset_spec == dataset_spec:
                    dataset = (
                        db.query(Dataset)
                        .filter(Dataset.file_path == dataset_path)
                        .first()
                    )
                    filtered_list.append(dataset)
            return filtered_list
    except Exception as e:
        logger.exception("Error filtering datasets: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while filtering datasets",
        )


@router.delete("/{predict_name}")
@inject
async def delete_prediction(
    predict_name: str,
    config: dict = Depends(lambda: di["config"]),
):
    """
    Deletes a prediction file based on the provided predict_name.

    Parameters
    ----------
    predict_name : str
        The name of the prediction file to delete.

    Raises
    ------
    HTTPException
        If the file cannot be found or deleted.
    """
    logger.debug("Deleting prediction file with name %s", predict_name)
    predict_path = os.path.join(config["DATASETS_PATH"], "predictions", predict_name)
    try:
        if os.path.exists(predict_path):
            os.remove(predict_path)
            logger.debug("File %s deleted successfully", predict_path)
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found",
            )
    except Exception as e:
        logger.exception("Error deleting file %s: %s", predict_name, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while deleting the prediction file",
        )


@router.patch("/{predict_name}")
@inject
async def rename_prediction(
    predict_name: str,
    request: RenameRequest,
    config: dict = Depends(lambda: di["config"]),
):
    """
    Renames a prediction file based on the provided predict_name.

    Parameters
    ----------
    predict_name : str
        The current name of the prediction file.
    new_name : str
        The new name for the prediction file.

    Raises
    ------
    HTTPException
        If the file cannot be found or renamed.
    """
    new_name = f"{request.new_name}.json "
    logger.debug("Renaming prediction file from %s to %s", predict_name, new_name)
    predict_path = os.path.join(config["DATASETS_PATH"], "predictions", predict_name)
    new_path = os.path.join(config["DATASETS_PATH"], "predictions", new_name)

    try:
        if os.path.exists(predict_path):
            with open(predict_path, "r") as json_file:
                data = json.load(json_file)
            data["metadata"]["pred_name"] = new_name
            with open(predict_path, "w") as json_file:
                json.dump(data, json_file, indent=4)
            os.rename(predict_path, new_path)
            logger.debug(
                "File renamed from %s to %s successfully", predict_path, new_path
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found",
            )
    except Exception as e:
        logger.exception(
            "Error renaming file %s to %s: %s", predict_name, new_name, str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while renaming the prediction file",
        )
