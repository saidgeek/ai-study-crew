import os
import chromadb
from crewai_tools import BaseTool

from utils.embedding import EmbeddingUtil


class ChromaBDTool(BaseTool):
    name: str = "ChromaDB Search Tool"
    description: str = "Tool for searching embeddings in ChromaDB."

    def _run(self, topic: str) -> str:
        embeddings = EmbeddingUtil()

        base_dir = os.path.abspath(os.path.dirname(__file__))
        db_path = os.path.join(
            base_dir,
            "../../../.chroma",
        )
        db = chromadb.PersistentClient(path=db_path)
        collection = db.get_collection(name="study_documets")

        context = ""

        results = collection.query(
            query_embeddings=embeddings.embed_query(topic), n_results=5
        )

        if len(results["ids"]) > 0:
            for doc in results["documents"]:
                context += f"```\n{doc}\n```\n\n"

        return context
