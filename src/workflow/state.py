from enum import Enum
import operator
from typing import Annotated, TypedDict


class NextType(Enum):
    RUNNER = "runner"
    PROCESS_DOCUMENTS = "process_documents"
    GENERATE_STUDY_CONTENT = "generate_study_content"
    FINISHED = "FINISHED"


class Steps(Enum):
    PROCESS_DOCUMENTS = "process_documents"
    GENERATE_CONTENT = "generate_content"


class StudyState(TypedDict):
    session_id: str | None
    temary: list[str]
    completed_steps: Annotated[list[str], operator.add]
    documents: list[str] | None
    next_resource: int = 0
    next: str
