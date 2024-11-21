import logging
from typing import Any, Dict, List, Union

from fastapi import APIRouter, Depends, status
from fastapi.exceptions import HTTPException
from kink import di, inject
from pydantic import BaseModel as PydanticBaseModel
from sqlalchemy import exc
from sqlalchemy.orm.session import sessionmaker

from DashAI.back.dependencies.database.models import ConverterList, Dataset

logger = logging.getLogger(__name__)
router = APIRouter()


class ConverterParams(PydanticBaseModel):
    params: Dict[str, Union[str, int, float, bool]] = None
    scope: Dict[str, List[int]] = None

    def serialize(self) -> Dict[str, Any]:
        return {"params": self.params, "scope": self.scope}


class ConverterListParams(PydanticBaseModel):
    dataset_id: int
    converters: Dict[str, ConverterParams]


@router.post("/", status_code=status.HTTP_201_CREATED)
@inject
async def post_dataset_converter_list(
    params: ConverterListParams,
    session_factory: sessionmaker = Depends(lambda: di["session_factory"]),
):
    """Save a list of converters to apply to the dataset.

    Parameters
    ----------
    dataset_id : int
        ID of the dataset.
    converters : Dict[str, ConverterParams]
        A dictionary with the converters to apply to the dataset.
    session_factory : Callable[..., ContextManager[Session]]
        A factory that creates a context manager that handles a SQLAlchemy session.
        The generated session can be used to access and query the database.

    Returns
    -------
    dict
        A dictionary with the ID of the converter list.

    Raises
    ------
    HTTPException
        If the dataset is not found or if there is an internal database error.
    """
    with session_factory() as db:
        try:
            dataset = db.get(Dataset, params.dataset_id)
            if not dataset:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Dataset not found",
                )
            serialized_converters = {
                key: value.serialize() for key, value in params.converters.items()
            }

            converter_list = ConverterList(
                dataset_id=params.dataset_id,
                converters=serialized_converters,
            )

            db.add(converter_list)
            db.commit()
            db.refresh(converter_list)

            return converter_list

        except exc.SQLAlchemyError as e:
            logger.exception(e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal database error",
            ) from e


@router.get("/{converter_list_id}")
@inject
async def get_dataset_converter_list(
    converter_list_id: int,
    session_factory: sessionmaker = Depends(lambda: di["session_factory"]),
):
    """Get a converter list from the database.

    Parameters
    ----------
    converter_list_id : int
        ID of the converter list.
    session_factory : Callable[..., ContextManager[Session]]
        A factory that creates a context manager that handles a SQLAlchemy session.
        The generated session can be used to access and query the database.

    Returns
    -------
    ConverterList
        The converter list.

    Raises
    ------
    HTTPException
        If the converter list is not found or if there is an internal database error.
    """
    with session_factory() as db:
        try:
            converter_list = db.get(ConverterList, converter_list_id)
            if not converter_list:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Converter list not found",
                )

            return converter_list

        except exc.SQLAlchemyError as e:
            logger.exception(e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal database error",
            ) from e
