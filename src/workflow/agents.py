import os
import chromadb

from langchain_community.document_loaders import PyPDFLoader

from workflow.state import NextType, Steps, StudyState
from langchain_openai import ChatOpenAI
from utils.embedding import EmbeddingUtil


class WorkflowAgentsPrepare:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model="llama3:8B", base_url="http://localhost:11434/v1", api_key="NA"
        )
        self.embeddings = EmbeddingUtil()

        base_dir = os.path.abspath(os.path.dirname(__file__))
        db_path = os.path.join(
            base_dir,
            "../../.chroma",
        )
        self.db = chromadb.PersistentClient(path=db_path)
        self.collection = self.db.get_or_create_collection(name="study_documets")

    @staticmethod
    def runner_agent(state: StudyState):
        print("\nRunning RunnerAgent")

        print(
            Steps.PROCESS_DOCUMENTS.value not in state["completed_steps"],
            Steps.PROCESS_DOCUMENTS.value,
            state["completed_steps"],
        )

        if Steps.PROCESS_DOCUMENTS.value not in state["completed_steps"]:
            print("Returning next PROCESS_DOCUMENTS")
            return {"next": NextType.PROCESS_DOCUMENTS.value}
        elif Steps.GENERATE_CONTENT.value not in state["completed_steps"]:
            print("Returning next GENERATE_STUDY_CONTENT")
            return {"next": NextType.GENERATE_STUDY_CONTENT.value}

        return {"next": NextType.FINISHED.value}

    def process_documents_agent(self, state: StudyState):
        print("\nRunning ProcessDocumentsAgent")

        for doc_path in state["documents"]:
            file_name = os.path.basename(doc_path)
            print("file_name:", file_name)

            loader = PyPDFLoader(doc_path)
            docs = loader.load()

            persist_docs = self.collection.get(where={"filename": file_name})

            if len(persist_docs["ids"]) == 0:
                [contents, embeddings] = self.embeddings.embed_documents(docs)

                self.collection.add(
                    ids=[str(i + 1) for i in range(len(contents))],
                    documents=contents,
                    embeddings=embeddings,
                    metadatas=[{"filename": file_name} for _ in range(len(contents))],
                )

        return {
            "next": NextType.RUNNER.value,
            "completed_steps": [Steps.PROCESS_DOCUMENTS.value],
        }
