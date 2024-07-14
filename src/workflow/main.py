import os
import uuid
import yaml

from workflow.state import StudyState
from workflow.agents import WorkflowAgentsPrepare
from study.crew import StudyCrew
from langgraph.graph import StateGraph, END


def load_settings(context: str):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(
        base_dir,
        f"../../examples/{context}/settings.yml",
    )

    data: StudyState = StudyState()

    with open(path, "r") as file:
        settings = yaml.safe_load(file)

    if len(settings["resources"]) > 0:
        data["documents"] = [
            os.path.join(
                base_dir,
                f"../../examples/{context}/{resource}",
            )
            for resource in settings["resources"]
        ]

    data = {**data, **settings}

    return data


def edge_condition(state: StudyState):
    print("\n", state["next"])
    print(state)
    return state["next"]


def run():
    data = load_settings("historia-de-chile")

    data["session_id"] = "fec344ae-3fab-43d6-bede-ee8c897310fc"

    if "session_id" not in data or data["session_id"] is None:
        data["session_id"] = str(uuid.uuid4())

    agents = WorkflowAgentsPrepare()

    graph = StateGraph(StudyState)

    graph.add_node("runner_node", agents.runner_agent)
    graph.add_node("process_documents_node", agents.process_documents_agent)
    graph.add_node("generate_study_content_node", StudyCrew().crew)

    graph.add_edge("process_documents_node", "runner_node")
    graph.add_edge("generate_study_content_node", "runner_node")

    graph.add_conditional_edges(
        "runner_node",
        edge_condition,
        {
            "process_documents": "process_documents_node",
            "generate_study_content": "generate_study_content_node",
            "FINISHED": END,
        },
    )

    graph.set_entry_point("runner_node")
    chain = graph.compile()

    result = chain.invoke({**data})

    print(result)
