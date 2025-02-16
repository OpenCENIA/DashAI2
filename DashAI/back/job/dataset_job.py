# filepath: /c:/Users/sinenomine/Desktop/DashAI/python-envs/DashAI/DashAI/back/job/dataset_upload_job.py
import logging
import os
import shutil
from typing import Any, Dict

from kink import inject
from sqlalchemy import exc
from sqlalchemy.orm import Session
from sqlalchemy.orm.session import sessionmaker
from starlette.datastructures import UploadFile

from DashAI.back.api.api_v1.schemas.datasets_params import DatasetParams
from DashAI.back.api.utils import parse_params
from DashAI.back.dataloaders.classes.dashai_dataset import save_dataset
from DashAI.back.dependencies.database.models import Dataset, Run
from DashAI.back.dependencies.registry import ComponentRegistry
from DashAI.back.job.base_job import BaseJob, JobError

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class DatasetJob(BaseJob):
    """Job to handle dataset upload."""

    def set_status_as_delivered(self) -> None:
        """Set the status of the job as delivered."""
        print("Delivered")

    @inject
    def run(
        self,
        component_registry: ComponentRegistry = lambda di: di["component_registry"],
        session_factory: sessionmaker = lambda di: di["session_factory"],
        config: Dict[str, Any] = lambda di: di["config"],
    ) -> None:

        file = self.kwargs["file"]
        contenido = file.file.read()
        print(contenido)
        params = self.kwargs
        url: str = self.kwargs.get("url")
        file = self.kwargs.get("file")

        parsed_params = DatasetParams.model_validate(params)
        dataloader = component_registry[parsed_params.dataloader]["class"]()
        folder_path = config["DATASETS_PATH"] / parsed_params.name

        # create dataset path
        try:
            log.debug("Trying to create a new dataset path: %s", folder_path)
            folder_path.mkdir(parents=True)
        except FileExistsError as e:
            log.exception(e)
            raise JobError("A dataset with this name already exists") from e

        # save dataset
        try:
            log.debug("Storing dataset in %s", folder_path)
            new_dataset = dataloader.load_data(
                filepath_or_buffer=file if file is not None else url,
                temp_path=str(folder_path),
                params=parsed_params.model_dump(),
            )

            dataset_path = folder_path / "dataset"
            log.debug("Saving dataset in %s", str(dataset_path))
            save_dataset(new_dataset, dataset_path)

        except OSError as e:
            shutil.rmtree(folder_path, ignore_errors=True)
            log.exception(e)
            raise JobError("Failed to read file") from e

        with session_factory() as db:
            log.debug("Storing dataset metadata in database.")
            try:
                folder_path = os.path.realpath(folder_path)
                new_dataset = Dataset(
                    name=parsed_params.name,
                    file_path=folder_path,
                )
                db.add(new_dataset)
                db.commit()
                db.refresh(new_dataset)

            except exc.SQLAlchemyError as e:
                log.exception(e)
                raise JobError("Internal database error") from e

        log.debug("Dataset creation successfully finished.")
