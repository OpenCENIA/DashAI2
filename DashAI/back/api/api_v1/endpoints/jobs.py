import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Request, Response, status
from fastapi.exceptions import HTTPException
from kink import di, inject
from sqlalchemy.orm import sessionmaker

from DashAI.back.api.api_v1.schemas.job_params import JobParams
from DashAI.back.dependencies.job_queues import BaseJobQueue
from DashAI.back.dependencies.job_queues.base_job_queue import JobQueueError
from DashAI.back.dependencies.job_queues.job_queue import job_queue_loop
from DashAI.back.dependencies.registry import ComponentRegistry
from DashAI.back.job.base_job import BaseJob, JobError

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/start/")
async def start_job_queue(
    background_tasks: BackgroundTasks,
    stop_when_queue_empties: bool = False,
):
    """Start the job queue to begin processing the jobs inside the jobs queue.
    If the param stop_when_queue_empties is True, the loop stops when the job queue
    becomes empty.

    Parameters
    ----------
    stop_when_queue_empties: Optional bool
        boolean to set the behavior of the loop.

    Returns
    -------
    Response
        response with code 202 ACCEPTED
    """
    background_tasks.add_task(job_queue_loop, stop_when_queue_empties)
    return Response(status_code=status.HTTP_202_ACCEPTED)


@router.get("/")
@inject
async def get_jobs(
    job_queue: BaseJobQueue = Depends(lambda: di["job_queue"]),
):
    """Return all the jobs in the job queue.

    Parameters
    ----------
    job_queue : BaseJobQueue
        The current app job queue.

    Returns
    ----------
    List[dict]
        A list of dict containing the Jobs.
    """
    all_jobs = job_queue.to_list()
    return all_jobs


@router.get("/{job_id}")
@inject
async def get_job(
    job_id: int,
    job_queue: BaseJobQueue = Depends(lambda: di["job_queue"]),
):
    """Return the selected job from the job queue

    Parameters
    ----------
    job_id: int
        id of the Job to get.
    job_queue : BaseJobQueue
        The current app job queue.

    Returns
    ----------
    dict
        A dict containing the Job information.

    Raises
    ----------
    HTTPException
        If is not posible to get the job from the job queue.
    """
    try:
        job = job_queue.peek(job_id)
    except JobQueueError as e:
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        ) from e
    return job


@router.post("/", status_code=status.HTTP_201_CREATED)
@inject
async def enqueue_job(
    request: Request,
    session_factory: sessionmaker = Depends(lambda: di["session_factory"]),
    component_registry: ComponentRegistry = Depends(lambda: di["component_registry"]),
    job_queue: BaseJobQueue = Depends(lambda: di["job_queue"]),
):
    """Create a job and put it in the job queue.

    Parameters
    ----------
    request : Request
        FastAPI request object that can handle both JSON and FormData
    session_factory : sessionmaker
        Database session factory
    component_registry : ComponentRegistry
        Registry containing available components
    job_queue : BaseJobQueue
        The job queue instance

    Returns
    -------
    dict
        The created job
    """
    try:
        content_type = request.headers.get("content-type", "")

        # Handle FormData
        if "multipart/form-data" in content_type:
            form = await request.form()
            job_params = JobParams(
                job_type=form.get("job_type"),
                kwargs={key: value for key, value in form.items() if key != "job_type"},
            )
        # Handle JSON
        else:
            print("application/json")
            data = await request.json()
            job_params = JobParams(**data)

        with session_factory() as db:
            job_params.kwargs["db"] = db
            job: BaseJob = component_registry[job_params.job_type]["class"](
                **job_params.model_dump()
            )

            try:
                job.set_status_as_delivered()
            except JobError as e:
                logger.exception(e)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Job not delivered",
                ) from e

            try:
                job_queue.put(job)
            except JobQueueError as e:
                logger.exception(e)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Job not enqueued",
                ) from e

        return job

    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.delete("/")
@inject
async def cancel_job(
    job_id: int,
    job_queue: BaseJobQueue = Depends(lambda: di["job_queue"]),
):
    """Delete the job with id job_id from the job queue.

    Parameters
    ----------
    job_id : int
        id of the job to delete.
    job_queue : BaseJobQueue
        The current app job queue.

    Returns
    -------
    Response
        response with code 204 NO_CONTENT

    Raises
    ----------
    HTTPException
        If is not posible to get the job from the job queue.
    """
    try:
        job_queue.get(job_id)
    except JobQueueError as e:
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        ) from e
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.patch("/")
async def update_job():
    """Placeholder for job update.

    Raises
    ------
    HTTPException
        Always raises exception as it was intentionally not implemented.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Method not implemented"
    )
