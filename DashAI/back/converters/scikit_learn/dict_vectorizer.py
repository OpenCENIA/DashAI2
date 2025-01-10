from sklearn.feature_extraction import DictVectorizer as DictVectorizerOperation

from DashAI.back.converters.sklearn_wrapper import SklearnWrapper
from DashAI.back.core.schema_fields import (
    bool_field,
    schema_field,
    string_field,
)
from DashAI.back.core.schema_fields.base_schema import BaseSchema


class DictVectorizerSchema(BaseSchema):
    # dtype: schema_field(
    #     none_type(enum_field(["np.float32", "np.float64"])), # dtype
    #     np.float64,
    #     "The type of feature values.",
    # )  # type: ignore
    separator: schema_field(
        string_field(),
        "=",
        "The separator string used in feature names if `feature_names` is an iterable of iterables.",
    )  # type: ignore
    sparse: schema_field(
        bool_field(),
        True,
        "Whether to return a sparse matrix.",
    )  # type: ignore
    sort: schema_field(
        bool_field(),
        True,
        "Whether to sort the feature indices.",
    )  # type: ignore


class DictVectorizer(SklearnWrapper, DictVectorizerOperation):
    """Scikit-learn's DictVectorizer wrapper for DashAI."""

    SCHEMA = DictVectorizerSchema
    DESCRIPTION = "This transformer turns lists of mappings (dict-like objects) of feature names to feature values into Numpy arrays or scipy.sparse matrices for use with scikit-learn estimators."
