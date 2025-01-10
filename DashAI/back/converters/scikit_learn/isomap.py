from sklearn.manifold import Isomap as IsomapOperation

from DashAI.back.converters.sklearn_wrapper import SklearnWrapper
from DashAI.back.core.schema_fields import (
    enum_field,
    float_field,
    int_field,
    none_type,
    schema_field,
)
from DashAI.back.core.schema_fields.base_schema import BaseSchema


class IsomapSchema(BaseSchema):
    n_neighbors: schema_field(
        none_type(int_field(ge=1)),
        5,
        "Number of neighbors to consider for each point.",
    )  # type: ignore
    radius: schema_field(
        none_type(float_field(gt=0)),
        None,
        "Radius of the neighborhood to consider for each point. This parameter is mutually exclusive with n_neighbors.",
    )  # type: ignore
    n_components: schema_field(
        int_field(ge=1),
        2,
        "Number of coordinates for the manifold.",
    )  # type: ignore
    eigen_solver: schema_field(
        enum_field(["auto", "arpack", "dense"]),
        "auto",
        "The eigenvalue decomposition strategy to use. 'auto' will attempt to choose the most appropriate solver.",
    )  # type: ignore
    tol: schema_field(
        float_field(gt=0),
        0,
        "Convergence tolerance passed to arpack or lobpcg. not used if eigen_solver=='dense'.",
    )  # type: ignore
    max_iter: schema_field(
        int_field(gt=0),
        None,
        "Maximum number of iterations for the arpack solver. not used if eigen_solver=='dense'.",
    )  # type: ignore
    path_method: schema_field(
        enum_field(["auto", "FW", "D"]),
        "auto",
        "Method to use in finding shortest path.",
    )  # type: ignore
    neighbors_algorithm: schema_field(
        enum_field(["auto", "brute", "kd_tree", "ball_tree"]),
        "auto",
        "Algorithm to use for nearest neighbors search.",
    )  # type: ignore
    n_jobs: schema_field(
        none_type(int_field()),
        None,
        "The number of parallel jobs to run for neighbors search. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.",
    )  # type: ignore
    metric: schema_field(
        enum_field(
            [
                "cityblock",
                "cosine",
                "euclidean",
                "l1",
                "l2",
                "manhattan",
                "braycurtis",
                "canberra",
                "chebyshev",
                "correlation",
                "dice",
                "hamming",
                "jaccard",
                "kulsinski",
                "mahalanobis",
                "minkowski",
                "rogerstanimoto",
                "russellrao",
                "seuclidean",
                "sokalmichener",
                "sokalsneath",
                "sqeuclidean",
                "yule",
                "precomputed",
            ]
        ),  # str or callable
        "minkowski",
        "Metric to use when calculating distance between instances in a feature array.",
    )  # type: ignore
    p: schema_field(
        int_field(gt=0),
        2,
        "Parameter for the Minkowski metric from sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.",
    )  # type: ignore
    # metric_params: schema_field(
    #     none_type(dict), # dict
    #     None,
    #     "Additional keyword arguments for the metric function.",
    # )  # type: ignore


class Isomap(SklearnWrapper, IsomapOperation):
    """Scikit-learn's Isomap wrapper for DashAI."""

    SCHEMA = IsomapSchema
    DESCRIPTION = "Non-linear dimensionality reduction through Isometric Mapping."
