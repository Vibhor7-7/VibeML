"""
SQLAlchemy models and CRUD operations for experiment tracking.
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Database setup
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vibeml.db'))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{db_path}")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Experiment(Base):
    """SQLAlchemy model for experiments."""
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    dataset_id = Column(String(255), nullable=False)
    problem_type = Column(String(50), nullable=False)  # classification, regression, etc.
    target_column = Column(String(255), nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    runs = relationship("Run", back_populates="experiment", cascade="all, delete-orphan")


class Run(Base):
    """SQLAlchemy model for experiment runs."""
    __tablename__ = "runs"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    run_number = Column(Integer, nullable=False)
    model_id = Column(String(255), nullable=True, index=True)
    
    # Algorithm and configuration
    algorithm = Column(String(100), nullable=False)
    status = Column(String(50), default="pending", nullable=False)  # pending, running, completed, failed
    
    # Celery task tracking
    celery_task_id = Column(String(255), nullable=True, index=True)
    
    # Parameters and results stored as JSON
    hyperparameters = Column(JSON, nullable=True)
    training_metrics = Column(JSON, nullable=True)
    validation_metrics = Column(JSON, nullable=True)
    test_metrics = Column(JSON, nullable=True)
    
    # Training details
    training_duration_seconds = Column(Float, nullable=True)
    dataset_size = Column(Integer, nullable=True)
    feature_count = Column(Integer, nullable=True)
    
    # Model artifacts
    model_path = Column(String(500), nullable=True)
    model_size_bytes = Column(Integer, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="runs")
    hyperparams = relationship("Hyperparameter", back_populates="run", cascade="all, delete-orphan")


class Hyperparameter(Base):
    """SQLAlchemy model for hyperparameters."""
    __tablename__ = "hyperparameters"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)
    parameter_name = Column(String(255), nullable=False)
    parameter_value = Column(Text, nullable=False)  # Store as string, parse as needed
    parameter_type = Column(String(50), nullable=False)  # int, float, str, bool, list, dict
    
    # Relationships
    run = relationship("Run", back_populates="hyperparams")


# Create tables
Base.metadata.create_all(bind=engine)


class ExperimentStore:
    """CRUD operations for experiment tracking."""
    
    def __init__(self, db: Session = None):
        self.db = db or SessionLocal()
    
    def create_experiment(
        self,
        name: str,
        dataset_id: str,
        problem_type: str,
        target_column: str,
        description: str = None
    ) -> Experiment:
        """Create a new experiment."""
        experiment = Experiment(
            name=name,
            description=description,
            dataset_id=dataset_id,
            problem_type=problem_type,
            target_column=target_column
        )
        self.db.add(experiment)
        self.db.commit()
        self.db.refresh(experiment)
        return experiment
    
    def get_experiment(self, experiment_id: int) -> Optional[Experiment]:
        """Get experiment by ID."""
        return self.db.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    def get_experiments(self, skip: int = 0, limit: int = 100) -> List[Experiment]:
        """Get all experiments with pagination."""
        return self.db.query(Experiment).offset(skip).limit(limit).all()
    
    def update_experiment(self, experiment_id: int, **kwargs) -> Optional[Experiment]:
        """Update experiment."""
        experiment = self.get_experiment(experiment_id)
        if experiment:
            for key, value in kwargs.items():
                if hasattr(experiment, key):
                    setattr(experiment, key, value)
            experiment.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(experiment)
        return experiment
    
    def delete_experiment(self, experiment_id: int) -> bool:
        """Delete experiment and all its runs."""
        experiment = self.get_experiment(experiment_id)
        if experiment:
            self.db.delete(experiment)
            self.db.commit()
            return True
        return False
    
    def create_run(
        self,
        experiment_id: int,
        algorithm: str,
        hyperparameters: Dict[str, Any] = None,
        model_id: str = None
    ) -> Run:
        """Create a new run for an experiment."""
        # Get the next run number for this experiment
        last_run = (
            self.db.query(Run)
            .filter(Run.experiment_id == experiment_id)
            .order_by(Run.run_number.desc())
            .first()
        )
        next_run_number = (last_run.run_number + 1) if last_run else 1
        
        run = Run(
            experiment_id=experiment_id,
            run_number=next_run_number,
            model_id=model_id,
            algorithm=algorithm,
            hyperparameters=hyperparameters or {},
            started_at=datetime.utcnow()
        )
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run
    
    def get_run(self, run_id: int) -> Optional[Run]:
        """Get run by ID."""
        return self.db.query(Run).filter(Run.id == run_id).first()
    
    def get_runs_by_experiment(self, experiment_id: int) -> List[Run]:
        """Get all runs for an experiment."""
        return self.db.query(Run).filter(Run.experiment_id == experiment_id).all()
    
    def update_run_status(self, run_id: int, status: str, error_message: str = None) -> Optional[Run]:
        """Update run status."""
        run = self.get_run(run_id)
        if run:
            run.status = status
            if error_message:
                run.error_message = error_message
            if status == "completed":
                run.completed_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(run)
        return run
    
    def update_run_metrics(
        self,
        run_id: int,
        training_metrics: Dict[str, float] = None,
        validation_metrics: Dict[str, float] = None,
        test_metrics: Dict[str, float] = None
    ) -> Optional[Run]:
        """Update run metrics."""
        run = self.get_run(run_id)
        if run:
            if training_metrics:
                run.training_metrics = training_metrics
            if validation_metrics:
                run.validation_metrics = validation_metrics
            if test_metrics:
                run.test_metrics = test_metrics
            self.db.commit()
            self.db.refresh(run)
        return run
    
    def update_run_model_info(
        self,
        run_id: int,
        model_path: str = None,
        model_size_bytes: int = None,
        training_duration_seconds: float = None
    ) -> Optional[Run]:
        """Update run model information."""
        run = self.get_run(run_id)
        if run:
            if model_path:
                run.model_path = model_path
            if model_size_bytes:
                run.model_size_bytes = model_size_bytes
            if training_duration_seconds:
                run.training_duration_seconds = training_duration_seconds
            self.db.commit()
            self.db.refresh(run)
        return run
    
    def get_best_run(self, experiment_id: int, metric: str = "accuracy") -> Optional[Run]:
        """Get the best run for an experiment based on a metric."""
        runs = self.get_runs_by_experiment(experiment_id)
        best_run = None
        best_score = float('-inf')
        
        for run in runs:
            if run.validation_metrics and metric in run.validation_metrics:
                score = run.validation_metrics[metric]
                if score > best_score:
                    best_score = score
                    best_run = run
        
        return best_run
    
    def get_experiment_summary(self, experiment_id: int) -> Dict[str, Any]:
        """Get summary statistics for an experiment."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None
        
        runs = self.get_runs_by_experiment(experiment_id)
        
        summary = {
            "experiment": {
                "id": experiment.id,
                "name": experiment.name,
                "description": experiment.description,
                "dataset_id": experiment.dataset_id,
                "problem_type": experiment.problem_type,
                "target_column": experiment.target_column,
                "created_at": experiment.created_at.isoformat(),
                "updated_at": experiment.updated_at.isoformat()
            },
            "runs": {
                "total_runs": len(runs),
                "completed_runs": len([r for r in runs if r.status == "completed"]),
                "failed_runs": len([r for r in runs if r.status == "failed"]),
                "running_runs": len([r for r in runs if r.status == "running"]),
                "pending_runs": len([r for r in runs if r.status == "pending"])
            },
            "algorithms_used": list(set([r.algorithm for r in runs])),
            "best_run": None
        }
        
        # Find best run based on validation accuracy or loss
        best_run = self.get_best_run(experiment_id, "accuracy") or self.get_best_run(experiment_id, "f1_score")
        if best_run:
            summary["best_run"] = {
                "run_id": best_run.id,
                "run_number": best_run.run_number,
                "algorithm": best_run.algorithm,
                "validation_metrics": best_run.validation_metrics,
                "training_duration": best_run.training_duration_seconds
            }
        
        return summary
    
    def close(self):
        """Close database session."""
        self.db.close()


# Dependency for FastAPI
def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_experiment_store(db: Session = None):
    """Get ExperimentStore instance."""
    return ExperimentStore(db)
