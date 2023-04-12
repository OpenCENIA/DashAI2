import logging
from typing import Union

from fastapi import APIRouter, Depends, Response, status
from fastapi.exceptions import HTTPException
from sqlalchemy import exc
from sqlalchemy.orm import Session

from DashAI.back.api.deps import get_db
from DashAI.back.database.models import Experiment

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def get_experiments(db: Session = Depends(get_db)):
    """
    Returns all the available experiments in the database.

    Returns
    -------
    List[JSON]
        List of dataset JSONs
    """

    try:
        all_experiments = db.query(Experiment).all()
        if not all_experiments:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="No experiments found"
            )
    except exc.SQLAlchemyError as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal database error",
        )
    return all_experiments


@router.get("/{experiment_id}")
async def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    """
    Returns the experiment with id experiment_id from the database.

    Parameters
    ----------
    experiment_id : int
        id of the experiment to query.

    Returns
    -------
    JSON
        JSON with the specified experiment id.
    """
    try:
        experiment = db.get(Experiment, experiment_id)
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found"
            )
    except exc.SQLAlchemyError as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal database error",
        )
    return experiment


@router.post("/", status_code=status.HTTP_201_CREATED)
async def upload_experiment(
    dataset_id: int,
    task_name: str,
    db: Session = Depends(get_db),
):
    """
    Endpoint to create experiments.

    Parameters
    ----------
    dataset_id : int
        Id of the Dataset linked to the experiment.
    task_name : str
        Name of the Task linked to the experiment.

    Returns
    -------
    JSON
        JSON with the new experiment on the database
    """
    try:
        experiment = Experiment(dataset_id=dataset_id, task_name=task_name)
        db.add(experiment)
        db.commit()
        db.refresh(experiment)
        return experiment
    except exc.SQLAlchemyError as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal database error",
        )


@router.delete("/{experiment_id}")
async def delete_experiment(experiment_id: int, db: Session = Depends(get_db)):
    """
    Deletes the experiment with id experiment_id from the database.

    Parameters
    ----------
    experiment_id : int
        id of the experiment to delete.

    Returns
    -------
    Response with code 204 NO_CONTENT
    """
    try:
        experiment = db.get(Experiment, experiment_id)
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found"
            )
        db.delete(experiment)
        db.commit()
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except exc.SQLAlchemyError as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal database error",
        )


@router.patch("/{experiment_id}")
async def update_dataset(
    experiment_id: int,
    db: Session = Depends(get_db),
    dataset_id: Union[int, None] = None,
    task_name: Union[str, None] = None,
):
    """
    Updates the experiment information with id experiment_id from the database.

    Parameters
    ----------
    experiment_id : int
        id of the experiment to delete.

    Returns
    -------
    JSON
        JSON containing the updated record
    """
    try:
        experiment = db.get(Experiment, experiment_id)
        if dataset_id:
            setattr(experiment, "name", dataset_id)
        if task_name:
            setattr(experiment, "task_name", task_name)
        if dataset_id or task_name:
            db.commit()
            db.refresh(experiment)
            return experiment
        else:
            raise HTTPException(
                status_code=status.HTTP_304_NOT_MODIFIED, detail="Record not modified"
            )
    except exc.SQLAlchemyError as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal database error",
        )
