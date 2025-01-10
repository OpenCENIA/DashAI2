from sklearn.preprocessing import KernelCenterer as KernelCentererOperation

from DashAI.back.converters.sklearn_wrapper import SklearnWrapper
from DashAI.back.core.schema_fields.base_schema import BaseSchema


class KernelCentererSchema(BaseSchema):
    pass


class KernelCenterer(SklearnWrapper, KernelCentererOperation):
    """Scikit-learn's KernelCenterer wrapper for DashAI."""

    SCHEMA = KernelCentererSchema
    DESCRIPTION = "Center a kernel matrix."
