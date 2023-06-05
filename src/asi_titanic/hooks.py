from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline.node import Node

from typing import Dict, Any
import mlflow
import mlflow.sklearn



class ModelTrackingHooks:
    """Namespace for grouping all model-tracking hooks with MLflow together."""

    @hook_impl
    def before_pipeline_run(self, run_params: Dict[str, Any]) -> None:
        """Hook implementation to start an MLflow run
        with the session_id of the Kedro pipeline run.
        """
        mlflow.start_run(run_name=run_params["session_id"])
        mlflow.log_params(run_params)
        mlflow.log_metric("score", 100)
        mlflow.sklearn.autolog()

    @hook_impl
    def after_pipeline_run(self) -> None:
        """Hook implementation to end the MLflow run
        after the Kedro pipeline finishes.
        """
        mlflow.end_run()
