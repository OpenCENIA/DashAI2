import logging

from fastapi import APIRouter, Depends, HTTPException, status
from kink import di
from sqlalchemy import exc
from sqlalchemy.orm import sessionmaker

from DashAI.back.api.api_v1.schemas.generative_session_params import (
    GenerativeSessionParams,
)
from DashAI.back.dependencies.database.models import GenerativeModel, GenerativeSession

router = APIRouter()
log = logging.getLogger(__name__)


@router.post("/", status_code=status.HTTP_201_CREATED)
async def upload_generative_session(
    params: GenerativeSessionParams,
    session_factory: sessionmaker = Depends(lambda: di["session_factory"]),
):
    """Create a new generative session.

    Parameters
    ----------
    params : GenerativesessionParams
        The parameters of the new generative session, which includes the model name,
        task name, parameters, session name and description.
    session_factory : Callable[..., ContextManager[Session]]
        A factory that creates a context manager that handles a SQLAlchemy session.
        The generated session can be used to access and query the database.

    Returns
    -------
    dict
        A dictionary with the new generative session on the database

    Raises
    ------
    HTTPException
        If there's an internal database error.
    """

    with session_factory() as db:
        try:
            model: GenerativeModel | None = (
                db.query(GenerativeModel).filter_by(name=params.model_name).one()
            )
            if not model:
                model = GenerativeModel(
                    name=params.model_name,
                    task_name=params.task_name,
                )
                db.add(model)
                db.commit()
                db.refresh(model)
            model_id = model.id

        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found",
            )

        try:
            session = GenerativeSession(
                model_id=model_id,
                parameters=params.parameters,
                name=params.name,
                description=params.description,
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            return session
        except exc.SQLAlchemyError as e:
            log.exception(e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal database error",
            ) from e
