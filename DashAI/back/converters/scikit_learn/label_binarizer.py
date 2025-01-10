from sklearn.preprocessing import LabelBinarizer as LabelBinarizerOperation

from DashAI.back.converters.sklearn_wrapper import SklearnWrapper
from DashAI.back.core.schema_fields import (
    bool_field,
    int_field,
    schema_field,
)
from DashAI.back.core.schema_fields.base_schema import BaseSchema


class LabelBinarizerSchema(BaseSchema):
    neg_label: schema_field(
        int_field(),
        0,
        "Value with which negative labels must be encoded.",
    )  # type: ignore
    pos_label: schema_field(
        int_field(),
        1,
        "Value with which positive labels must be encoded.",
    )  # type: ignore
    sparse_output: schema_field(
        bool_field(),
        False,
        "True if the returned array from transform is desired to be in sparse CSR format.",
    )  # type: ignore


class LabelBinarizer(SklearnWrapper, LabelBinarizerOperation):
    """Scikit-learn's LabelBinarizer wrapper for DashAI."""

    SCHEMA = LabelBinarizerSchema
    DESCRIPTION = "Binarize labels in a one-vs-all fashion."
