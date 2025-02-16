from typing import Any, Dict, Literal, Optional

from fastapi import UploadFile
from pydantic import BaseModel, ConfigDict, Field


class JobParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    job_type: Literal["ModelJob", "ExplainerJob", "DatasetJob"]
    file: Optional[UploadFile]
    kwargs: str
