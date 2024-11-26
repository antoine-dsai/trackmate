from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient
import os
from dotenv import load_dotenv
import tempfile
from concurrent.futures import ThreadPoolExecutor
import asyncio
import logging

# Load environment variables
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Initialize MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Initialize FastAPI app
app = FastAPI(title="TrackMate Backend")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class ExperimentCreate(BaseModel):
    name: str
    description: str | None = None

class RunCreate(BaseModel):
    run_name: str | None = None
    nested: bool = False

class Param(BaseModel):
    key: str
    value: str

class Metric(BaseModel):
    key: str
    value: float

# Response models
class ExperimentResponse(BaseModel):
    experiment_id: str
    name: str
    artifact_location: str

class RunResponse(BaseModel):
    run_id: str
    status: str
    experiment_id: str
    start_time: int
    end_time: int | None
    parameters: dict
    metrics: dict
    tags: dict

class ArtifactResponse(BaseModel):
    message: str
    filename: str

# Utility function to handle blocking MLflow calls asynchronously
def blocking_mlflow_call(func, *args, **kwargs):
    with ThreadPoolExecutor() as pool:
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(pool, lambda: func(*args, **kwargs))

# Experiment endpoints
@app.post("/experiments/")
async def create_experiment(exp: ExperimentCreate):
    client = MlflowClient()
    existing_experiment = client.get_experiment_by_name(exp.name)
    if existing_experiment:
        return {"experiment_id": existing_experiment.experiment_id, "name": exp.name}

    experiment_id = await blocking_mlflow_call(mlflow.create_experiment, exp.name, exp.description)
    return {"experiment_id": experiment_id, "name": exp.name}

@app.get("/experiments/", response_model=list[ExperimentResponse])
async def list_experiments():
    experiments = await blocking_mlflow_call(MlflowClient().list_experiments)
    exp_list = [
        ExperimentResponse(
            experiment_id=exp.experiment_id,
            name=exp.name,
            artifact_location=exp.artifact_location
        )
        for exp in experiments
    ]
    return exp_list

# Run endpoints
@app.post("/experiments/{experiment_id}/runs/")
async def start_run(experiment_id: str, run: RunCreate):
    client = MlflowClient()
    experiment = client.get_experiment(experiment_id)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run.run_name, nested=run.nested) as active_run:
        return {"run_id": active_run.info.run_id, "status": active_run.info.status}

@app.post("/runs/{run_id}/params/")
async def log_param(run_id: str, param: Param):
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param(param.key, param.value)
    mlflow.end_run()
    return {"message": f"Parameter '{param.key}' logged successfully."}

@app.post("/runs/{run_id}/metrics/")
async def log_metric(run_id: str, metric: Metric):
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric(metric.key, metric.value)
    return {"message": f"Metric '{metric.key}' logged successfully."}

# Artifact endpoints
@app.post("/runs/{run_id}/artifacts/", response_model=ArtifactResponse)
async def log_artifact(run_id: str, file: UploadFile = File(...)):
    with mlflow.start_run(run_id=run_id):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name

        mlflow.log_artifact(temp_file_path)

        # Clean up the temporary file after logging
        os.remove(temp_file_path)

    return ArtifactResponse(message=f"Artifact '{file.filename}' logged successfully.", filename=file.filename)

# Get a specific run
@app.get("/experiments/{experiment_id}/runs/{run_id}/", response_model=RunResponse)
async def get_run(experiment_id: str, run_id: str):
    run = await blocking_mlflow_call(mlflow.get_run, run_id)
    run_data = RunResponse(
        run_id=run.info.run_id,
        experiment_id=run.info.experiment_id,
        status=run.info.status,
        start_time=run.info.start_time,
        end_time=run.info.end_time,
        parameters=run.data.params,
        metrics=run.data.metrics,
        tags=run.data.tags,
    )
    return run_data
