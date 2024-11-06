from sklearn.kernel_approximation import Nystroem as NystroemOperation

from DashAI.back.core.schema_fields import (
    schema_field,
    int_field,
    float_field,
    none_type,
    string_field,
)

from DashAI.back.core.schema_fields.base_schema import BaseSchema
from DashAI.back.converters.scikit_learn.sklearn_like_converter import (
    SklearnLikeConverter,
)


class NystroemSchema(BaseSchema):
    kernel: schema_field(
        none_type(string_field()),  # str or callable
        "rbf",
        "The kernel to use for the approximation.",
    )  # type: ignore
    gamma: schema_field(
        none_type(float_field(gt=0)),
        None,
        "The gamma parameter for the RBF, laplacian, polynomial, exponential chi2 and sigmoid kernels.",
    )  # type: ignore
    coef0: schema_field(
        none_type(float_field()),
        None,
        "The coef0 parameter for the polynomial and sigmoid kernels.",
    )  # type: ignore
    degree: schema_field(
        none_type(float_field(ge=1)),
        None,
        "The degree of the polynomial kernel.",
    )  # type: ignore
    # kernel_params: schema_field(
    #     none_type(dict), # dict
    #     None,
    #     "Additional parameters (keyword arguments) for the kernel function.",
    # )  # type: ignore
    n_components: schema_field(
        int_field(ge=1),
        100,
        "The number of features to construct.",
    )  # type: ignore
    random_state: schema_field(
        none_type(int_field()),  # int, RandomState instance or None
        None,
        "The seed of the pseudo random number generator to use when shuffling the data.",
    )  # type: ignore
    n_jobs: schema_field(
        none_type(int_field()),
        None,
        "Number of parallel jobs to run.",
    )  # type: ignore


class Nystroem(SklearnLikeConverter, NystroemOperation):
    """Scikit-learn's Nystroem wrapper for DashAI."""

    SCHEMA = NystroemSchema
    DESCRIPTION = "Approximates the feature map of an RBF kernel by Monte Carlo approximation of its Fourier transform."
