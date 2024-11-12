import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List

import numpy as np
from fastapi import Depends, status
from fastapi.exceptions import HTTPException
from kink import di, inject
from sqlalchemy import exc
from sqlalchemy.orm import Session, sessionmaker

from DashAI.back.dataloaders.classes.dashai_dataset import DashAIDataset, load_dataset
from DashAI.back.dependencies.database.models import Dataset, Experiment, Run
from DashAI.back.dependencies.registry import ComponentRegistry
from DashAI.back.job.base_job import BaseJob, JobError
from DashAI.back.models.base_model import BaseModel

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class PredictJob(BaseJob):
    """PredictJob class to run the prediction."""

    def set_status_as_delivered(self) -> None:
        """Set the status of the job as delivered."""
        run_id: int = self.kwargs["run_id"]
        db: Session = self.kwargs["db"]

        run: Run = db.get(Run, run_id)
        if not run:
            raise JobError(f"Run {run_id} does not exist in DB.")
        try:
            run.set_status_as_delivered()
            db.commit()
        except exc.SQLAlchemyError as e:
            log.exception(e)
            raise JobError(
                "Internal database error",
            ) from e

    @inject
    def run(
        self,
        component_registry: ComponentRegistry = lambda di: di["component_registry"],
        session_factory: sessionmaker = Depends(lambda: di["session_factory"]),
        config=lambda di: di["config"],
    ) -> List[Any]:

        run_id: int = self.kwargs["run_id"]
        id: int = self.kwargs["id"]
        db: Session = self.kwargs["db"]
        try:
            run: Run = db.get(Run, run_id)
            if not run:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Run not found"
                )

            exp: Experiment = db.get(Experiment, run.experiment_id)
            if not exp:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Experiment not found",
                )
            dataset: Dataset = db.get(Dataset, id)
            if not dataset:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Dataset not found",
                )
        except exc.SQLAlchemyError as e:
            log.exception(e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal database error",
            ) from e
        try:
            loaded_dataset: DashAIDataset = load_dataset(
                str(Path(f"{dataset.file_path}/dataset/"))
            )["train"]
        except Exception as e:
            log.exception(e)
            raise JobError(
                "Can not load dataset from path {dataset.file_path}/dataset/"
            ) from e
        try:
            model = component_registry[run.model_name]["class"]
        except Exception as e:
            log.exception(e)
            raise JobError(f"Model {run.model_name} not found in the registry") from e

        try:
            trained_model: BaseModel = model.load(run.run_path)
            y_pred_proba = np.array(
                trained_model.predict(loaded_dataset.select_columns(exp.input_columns))
            )
            y_pred = np.argmax(y_pred_proba, axis=1)
        except Exception as e:
            log.exception(e)
            raise JobError(
                "Model prediction failed",
            ) from e
        try:
            new_dataset_name = f"{dataset.name}_pred"
            existing_dataset = (
                db.query(Dataset).filter_by(name=new_dataset_name).first()
            )
            if existing_dataset:
                log.info(
                    f"Dataset {new_dataset_name} already exists. Skipping creation."
                )
                return [existing_dataset]

            new_dataset: DashAIDataset = loaded_dataset.add_column("prediction", y_pred)
            path = str(Path(f"{config['DATASETS_PATH']}/{new_dataset_name}/"))
            new_dataset.save_to_disk(os.path.join(path, "dataset/train/"))
        except Exception as e:
            log.exception(e)
            raise JobError("Can not save the prediction dataset") from e

        new_dataset = Dataset(
            name=new_dataset_name,
            for_prediction=True,
            prediction_status=True,
            model_name=run.name,
            last_modified=datetime.now(),
            file_path=path,
        )

        # Save the 'dataset_dict.json' file to indicate that the only split is 'train'
        with open(
            os.path.join(f"{new_dataset.file_path}/dataset/", "dataset_dict.json"),
            "w",
            encoding="utf-8",
        ) as datasetdict_info_file:
            print("Saving dataset_dict.json")
            json.dump(
                {"splits": ["train"]},
                datasetdict_info_file,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        try:
            db.add(new_dataset)
            db.commit()
            db.refresh(new_dataset)
        except exc.SQLAlchemyError as e:
            log.exception(e)
            raise JobError("Connection with the database failed") from e
