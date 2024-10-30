from sklearn.decomposition import IncrementalPCA as IncrementalPCAOperation

from DashAI.back.core.schema_fields import (
    schema_field,
    int_field,
    none_type,
    bool_field,
)

from DashAI.back.core.schema_fields.base_schema import BaseSchema
from DashAI.back.converters.scikit_learn.sklearn_like_converter import (
    SklearnLikeConverter,
)


class IncrementalPCASchema(BaseSchema):
    n_components: schema_field(
        none_type(int_field(ge=1)),
        None,
        "Number of components to keep.",
    )  # type: ignore
    whiten: schema_field(
        bool_field(),
        False,
        "When True (False by default) the components_ vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.",
    )  # type: ignore
    copy: schema_field(
        bool_field(),
        True,
        "If False, data passed to fit are overwritten and running fit(X).transform(X) will not yield the expected results, use fit_transform(X) instead.",
    )  # type: ignore
    batch_size: schema_field(
        none_type(int_field(ge=1)),
        None,
        "The number of samples to use for each batch.",
    )  # type: ignore


class IncrementalPCA(SklearnLikeConverter, IncrementalPCAOperation):
    """Scikit-learn's IncrementalPCA wrapper for DashAI."""

    SCHEMA = IncrementalPCASchema
    DESCRIPTION = "Incremental principal components analysis (IPCA)."
