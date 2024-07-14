import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_openai import ChatOpenAI
from workflow.state import NextType, Steps, StudyState

from crewai_tools import SerperDevTool
from study.tools.chromadb import ChromaBDTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

os.environ["SERPER_API_KEY"] = "3c1a69627a51d3fee6c011c76e3d9196b16d0a47"


@CrewBase
class StudyCrew:
    """Study crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    llm = ChatOpenAI(
        model="llama3:8B", base_url="http://localhost:11434/v1", api_key="NA"
    )

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            tools=[
                SerperDevTool(),
            ],  # Example of custom tool, loaded on the beginning of file
            llm=self.llm,
            verbose=True,
        )

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"], agent=self.researcher())

    @crew
    def crew(self, state: StudyState):
        print("state:", state)

        result = Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=2,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        ).kickoff(inputs={"topic": state["temary"][0]})

        print("crew result:", result)

        return {
            "next": NextType.RUNNER.value,
            "completed_steps": [Steps.GENERATE_CONTENT.value],
        }
