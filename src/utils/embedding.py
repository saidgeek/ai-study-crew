from typing import List

from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from concurrent.futures import ThreadPoolExecutor


class EmbeddingUtil:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    def embed_documents(self, documents: List[Document]):
        splitter = CharacterTextSplitter(
            separator="\n\n", chunk_size=1000, chunk_overlap=200
        )
        chucked_docs = splitter.split_documents(documents)

        chunks = [cd.page_content for cd in chucked_docs]
        embed_chunks = self.embeddings.embed_documents(chunks)

        return chunks, embed_chunks

    def embed_query(self, query: str):
        emb_query = self.embeddings.embed_query(query)

        return emb_query
