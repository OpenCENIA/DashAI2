from datetime import datetime
from typing import List

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from DashAI.back.core.enums.status import RunStatus, UserStep


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "dataset"
    """
    Table to store all the information about a dataset.
    """
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    task_name: Mapped[str] = mapped_column(String, nullable=False)
    created: Mapped[DateTime] = mapped_column(DateTime, default=datetime.now)
    last_modified: Mapped[DateTime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    experiments: Mapped[List["Experiment"]] = relationship()


class Experiment(Base):
    __tablename__ = "experiment"
    """
    Table to store all the information about a model.
    """
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    task_name: Mapped[str] = mapped_column(String, nullable=False)
    step: Mapped[Enum] = mapped_column(Enum(UserStep), nullable=False)
    created: Mapped[DateTime] = mapped_column(DateTime, default=datetime.now)
    last_modified: Mapped[DateTime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )
    runs: Mapped[List["Run"]] = relationship()


class Run(Base):
    __tablename__ = "run"
    """
    Table to store all the information about a specific run of a model.
    """
    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment.id"))
    created: Mapped[DateTime] = mapped_column(DateTime, default=datetime.now)
    last_modified: Mapped[DateTime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )
    # model and parameters
    model_name: Mapped[str] = mapped_column(String)
    parameters: Mapped[JSON] = mapped_column(JSON)
    # metrics
    train_metrics: Mapped[JSON] = mapped_column(JSON)
    test_metrics: Mapped[JSON] = mapped_column(JSON)
    validation_metrics: Mapped[JSON] = mapped_column(JSON)
    # artifacts
    artifacts: Mapped[str] = mapped_column(JSON)
    # metadata
    run_name: Mapped[str] = mapped_column(String)
    run_description: Mapped[str] = mapped_column(String)
    status: Mapped[Enum] = mapped_column(Enum(RunStatus), nullable=False)
    start_time: Mapped[DateTime] = mapped_column(DateTime)
    end_time: Mapped[DateTime] = mapped_column(DateTime)
