from typing import Any, Dict, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class JobParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    job_type: Literal["ModelJob", "ExplainerJob", "DatasetJob"]
    kwargs: Dict[str, Any] = Field(default_factory=dict)
