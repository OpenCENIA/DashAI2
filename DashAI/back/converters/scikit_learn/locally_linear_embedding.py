from sklearn.manifold import LocallyLinearEmbedding as LocallyLinearEmbeddingOperation

from DashAI.back.converters.sklearn_wrapper import SklearnWrapper
from DashAI.back.core.schema_fields import (
    enum_field,
    float_field,
    int_field,
    none_type,
    schema_field,
)
from DashAI.back.core.schema_fields.base_schema import BaseSchema


class LocallyLinearEmbeddingSchema(BaseSchema):
    n_neighbors: schema_field(
        int_field(ge=1),
        5,
        "Number of neighbors to consider for each point.",
    )  # type: ignore
    n_components: schema_field(
        int_field(ge=1),
        2,
        "Number of coordinates for the manifold.",
    )  # type: ignore
    reg: schema_field(
        float_field(gt=0),
        1e-3,
        "Regularization constant, multiplies the trace of the local covariance matrix of the distances.",
    )  # type: ignore
    eigen_solver: schema_field(
        enum_field(["auto", "arpack", "dense"]),
        "auto",
        "The eigenvalue decomposition strategy to use. 'auto' will attempt to choose the most appropriate solver.",
    )  # type: ignore
    tol: schema_field(
        float_field(gt=0),
        1e-6,
        "Tolerance for ARPACK. 0 means machine precision.",
    )  # type: ignore
    max_iter: schema_field(
        int_field(gt=0),
        100,
        "Maximum number of iterations for the arpack solver.",
    )  # type: ignore
    method: schema_field(
        enum_field(["standard", "hessian", "modified", "ltsa"]),
        "standard",
        "Standard LLE algorithm will be used.",
    )  # type: ignore
    hessian_tol: schema_field(
        float_field(gt=0),
        1e-4,
        "Tolerance for Hessian eigenmapping method.",
    )  # type: ignore
    modified_tol: schema_field(
        float_field(gt=0),
        1e-12,
        "Tolerance for modified LLE method.",
    )  # type: ignore
    neighbors_algorithm: schema_field(
        enum_field(["auto", "brute", "kd_tree", "ball_tree"]),
        "auto",
        "Algorithm to use for nearest neighbors search.",
    )  # type: ignore
    random_state: schema_field(
        none_type(int_field()),  # int, RandomState instance or None
        None,
        "Pseudo-random number generator to control the generation of the random weights and random offset when fitting the training data. Pass an int for reproducible output across multiple function calls.",
    )  # type: ignore
    n_jobs: schema_field(
        none_type(int_field()),
        None,
        "The number of parallel jobs to run for neighbors search.",
    )  # type: ignore


class LocallyLinearEmbedding(SklearnWrapper, LocallyLinearEmbeddingOperation):
    """Scikit-learn's LocallyLinearEmbedding wrapper for DashAI."""

    SCHEMA = LocallyLinearEmbeddingSchema
    DESCRIPTION = "Locally Linear Embedding (LLE) seeks a lower-dimensional projection of the data that preserves distances within local neighborhoods."
