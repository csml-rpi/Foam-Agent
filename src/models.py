from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CreateCaseIn(BaseModel):
    user_prompt: str
    output_dir: Optional[str] = None


class CreateCaseOut(BaseModel):
    case_id: str
    case_dir: str


class PlanIn(BaseModel):
    case_id: str


class Subtask(BaseModel):
    file: str
    folder: str


class PlanOut(BaseModel):
    plan: List[Subtask]
    case_info: Dict


class GenerateFileIn(BaseModel):
    case_id: str
    file: str
    folder: str
    write: bool = True
    overwrite: bool = True


class GenerateFileOut(BaseModel):
    content: str
    written_path: Optional[str] = None


class MeshIn(BaseModel):
    case_id: str
    mesh_config: Dict


class MeshOut(BaseModel):
    job_id: Optional[str] = None
    status: str


class HPCScriptIn(BaseModel):
    case_id: str
    hpc_config: Dict


class HPCScriptOut(BaseModel):
    script_content: str
    script_path: str


class RunIn(BaseModel):
    case_id: str
    environment: str  # "local" | "hpc"
    extra: Optional[Dict] = None


class RunOut(BaseModel):
    job_id: Optional[str]
    status: str  # "submitted" | "completed" | "failed"


class JobStatusIn(BaseModel):
    job_id: str


class JobStatusOut(BaseModel):
    status: str
    details: Optional[Dict] = None


class LogsIn(BaseModel):
    case_id: str
    job_id: Optional[str] = None


class LogsOut(BaseModel):
    logs: Dict[str, str]


class ApplyFixIn(BaseModel):
    case_id: str
    foamfiles: Optional[Any] = None  # FoamPydantic object
    error_logs: List[str] = []
    review_analysis: str = ""
    user_requirement: str = ""
    dir_structure: Optional[Dict] = None


class ApplyFixOut(BaseModel):
    status: str
    written: List[str]
    updated_dir_structure: Optional[Dict] = None
    updated_foamfiles: Optional[Any] = None  # FoamPydantic object
    cleared_error_logs: List[str] = []


class VisualizationIn(BaseModel):
    case_id: str
    quantity: str
    extra: Optional[Dict] = None


class VisualizationOut(BaseModel):
    job_id: Optional[str]
    artifacts: List[str]


