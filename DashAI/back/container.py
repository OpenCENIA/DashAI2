import logging
from typing import Dict

from kink import Container, di

from DashAI.back.dataloaders import (
    CSVDataLoader,
    ExcelDataLoader,
    ImageDataLoader,
    JSONDataLoader,
)
from DashAI.back.dependencies.database import setup_sqlite_db
from DashAI.back.dependencies.job_queues import SimpleJobQueue
from DashAI.back.dependencies.registry import ComponentRegistry
from DashAI.back.explainability import (
    KernelShap,
    PartialDependence,
    PermutationFeatureImportance,
)
from DashAI.back.job import ExplainerJob, ModelJob
from DashAI.back.metrics import F1, MAE, RMSE, Accuracy, Bleu, Precision, Recall
from DashAI.back.models import (
    SVC,
    BagOfWordsTextClassificationModel,
    DecisionTreeClassifier,
    DistilBertTransformer,
    DummyClassifier,
    GradientBoostingR,
    HistGradientBoostingClassifier,
    KNeighborsClassifier,
    LinearRegression,
    LinearSVR,
    LogisticRegression,
    MLPRegression,
    OpusMtEnESTransformer,
    RandomForestClassifier,
    RandomForestRegression,
    RidgeRegression,
    ViTTransformer,
)
from DashAI.back.optimizers import (
    HyperOptOptimizer,
    OptunaOptimizer,
)
from DashAI.back.tasks import (
    ImageClassificationTask,
    RegressionTask,
    TabularClassificationTask,
    TextClassificationTask,
    TranslationTask,
)

logger = logging.getLogger(__name__)


INITIAL_COMPONENTS = [
    # Tasks
    TabularClassificationTask,
    TextClassificationTask,
    TranslationTask,
    ImageClassificationTask,
    RegressionTask,
    # Models
    SVC,
    DecisionTreeClassifier,
    DummyClassifier,
    GradientBoostingR,
    HistGradientBoostingClassifier,
    KNeighborsClassifier,
    LogisticRegression,
    MLPRegression,
    RandomForestClassifier,
    RandomForestRegression,
    DistilBertTransformer,
    ViTTransformer,
    OpusMtEnESTransformer,
    BagOfWordsTextClassificationModel,
    RidgeRegression,
    LinearSVR,
    LinearRegression,
    # Dataloaders
    CSVDataLoader,
    JSONDataLoader,
    ImageDataLoader,
    ExcelDataLoader,
    # Metrics
    F1,
    Accuracy,
    Precision,
    Recall,
    Bleu,
    MAE,
    RMSE,
    # Optimizers
    OptunaOptimizer,
    HyperOptOptimizer,
    # Jobs
    ExplainerJob,
    ModelJob,
    # Explainers
    KernelShap,
    PartialDependence,
    PermutationFeatureImportance,
]


def build_container(config: Dict[str, str]) -> Container:
    """
    Creates a dependency injection container for the application.

    Parameters
    ----------
    config : Dict[str, str]
        A dictionary containing configuration settings.

    Returns
    -------
    Container
        A dependency injection container instance populated with
        essential services. These services include:
            * 'config': The provided configuration dictionary.
            * Engine: The created SQLAlchemy engine for the SQLite database.
            * sessionmaker: A session factory for creating database sessions.
            * ComponentRegistry: The app component registry.
            * BaseJobQueue: The app job queue.
    """
    engine, session_factory = setup_sqlite_db(config)

    di["config"] = config
    di["engine"] = engine
    di["session_factory"] = session_factory
    di["component_registry"] = ComponentRegistry(initial_components=INITIAL_COMPONENTS)
    di["job_queue"] = SimpleJobQueue()

    return di
