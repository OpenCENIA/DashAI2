import logging

from fastapi import APIRouter, Depends, status
from fastapi.exceptions import HTTPException
from kink import di, inject
from sqlalchemy.orm import sessionmaker

from DashAI.back.dependencies.database.models import Dataset, Experiment, Run

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
@inject
async def get_prediction_tab(
    table: str,
    session_factory: sessionmaker = Depends(lambda: di["session_factory"]),
):
    """
    Fetches data based on the specified table parameter.

    Parameters
    ----------
    table : str
        A parameter that determines the type of data to fetch.
        - If 'PredictionTable', returns prediction-related data.
        - If 'SelectModelStep', returns model selection-related data.

    Raises
    ------
    HTTPException
        "No data found" if no data is found.

    Returns
    -------
    dict
        A dictionary containing the fetched data based on the table parameter.
    """
    with session_factory() as db:
        if table == "PredictionTable/":
            query_results = (
                db.query(
                    Experiment.task_name,
                    Run.model_name.label("run_type"),
                    Dataset.name.label("dataset_name"),
                    Dataset.id.label("dataset_id"),
                    Dataset.model_name.label("dataset_model_name"),
                    Dataset.last_modified,
                )
                .filter(Dataset.prediction_status)
                .all()
            )

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

        elif table == "SelectModelStep/":
            query_results = (
                db.query(
                    Experiment.id.label("experiment_id"),
                    Experiment.name.label("experiment_name"),
                    Experiment.created,
                    Experiment.task_name,
                    Run.name.label("run_name"),
                    Run.model_name,
                    Dataset.name.label("dataset_name"),
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
                }
                for result in query_results
            ]
            return prediction_data


@router.delete("/")
@inject
async def delete_prediction():
    """Placeholder for prediction delete.

    Raises
    ------
    HTTPException
        Always raises exception as it was intentionally not implemented.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Method not implemented"
    )


@router.patch("/")
@inject
async def update_prediction():
    """Placeholder for prediction update.

    Raises
    ------
    HTTPException
        Always raises exception as it was intentionally not implemented.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Method not implemented"
    )
