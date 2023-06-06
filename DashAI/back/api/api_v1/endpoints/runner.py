import logging
from typing import Dict

from fastapi import APIRouter, Depends, Response, status
from fastapi.exceptions import HTTPException
from sqlalchemy import exc
from sqlalchemy.orm import Session

from DashAI.back.api.api_v1.endpoints.components import _intersect_component_lists
from DashAI.back.api.deps import get_db
from DashAI.back.core.config import component_registry, settings
from DashAI.back.database.models import Dataset, Experiment, Run
from DashAI.back.dataloaders.classes.dashai_dataset import DashAIDataset, load_dataset

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/")
async def execute_run(run_id: int, db: Session = Depends(get_db)):
    """
    Train and evaluate the given run.

    Parameters
    ----------
    run_id : int
        id of the run to query, train and evaluate.

    """
    try:
        run: Run = db.get(Run, run_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Run not found"
            )
        exp: Experiment = db.get(Experiment, run.experiment_id)
        if not exp:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found"
            )
        dat: Dataset = db.get(Dataset, exp.dataset_id)
        if not dat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found"
            )
    except exc.SQLAlchemyError as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal database error",
        )
    run_instance = component_registry[run.model_name]["class"](**run.parameters)
    dataset_dict = load_dataset(f"{dat.file_path}/dataset")
    # Prepare dataset
    dataset_dict = component_registry[exp.task_name]["class"]().prepare_for_task(
        dataset_dict
    )
    # Format dataset: TBD move this to dataloader
    dataset_dict = run_instance.format_data(dataset_dict)
    # Training
    run.run_start()
    db.commit()
    run_instance.fit(dataset_dict["train"]["input"], dataset_dict["train"]["output"])
    # Evaluation
    metrics = _intersect_component_lists(
        component_registry.get_components_by_types(select="Metric"),
        component_registry.get_related_components(exp.task_name),
    )
    run.train_metrics: Dict[str, DashAIDataset] = {
        metric_name: metric_cls.score(
            dataset_dict["train"]["output"],
            run_instance.predict(dataset_dict["train"]["input"]),
        )
        for metric_name, metric_cls in metrics.items()
    }
    run.validation_metrics: Dict[str, DashAIDataset] = {
        metric_name: metric_cls.score(
            dataset_dict["validation"]["output"],
            run_instance.predict(dataset_dict["validation"]["input"]),
        )
        for metric_name, metric_cls in metrics.items()
    }
    run.test_metrics = {
        metric_name: metric_cls.score(
            dataset_dict["test"]["output"],
            run_instance.predict(dataset_dict["test"]["input"]),
        )
        for metric_name, metric_cls in metrics.items()
    }
    run.run_end()
    # Save changes
    run_path = f"{settings.USER_RUN_PATH}/{run.id}"
    run_instance.save(run_path)
    run.run_path = run_path
    db.commit()

    return Response(status_code=status.HTTP_202_ACCEPTED)
