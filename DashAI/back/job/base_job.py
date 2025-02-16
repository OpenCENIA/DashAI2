"""Base Job abstract class."""

from abc import ABCMeta, abstractmethod
from typing import Final


class BaseJob(metaclass=ABCMeta):
    """Abstract class for all Jobs."""

    TYPE: Final[str] = "Job"

    def __init__(self, **kwargs):
        """Constructor of the ModelJob class.

        Parameters
        ----------
        kwargs: dict
            dictionary containing the parameters of the job.
        """
        file = kwargs.pop("file", None)
        print(kwargs["file"].file.read())
        job_kwargs = kwargs.pop("kwargs", {})
        self.kwargs = {**kwargs, **job_kwargs, file: file}

    @abstractmethod
    def set_status_as_delivered(self) -> None:
        """Set the status of the job as delivered."""
        raise NotImplementedError

    @abstractmethod
    def run() -> None:
        """Run the job."""
        raise NotImplementedError


class JobError(Exception):
    """Exception raised when the job proccess fails."""
