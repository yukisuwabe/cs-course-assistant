from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings

class Retriever:
    """Manages document processing and vector store creation."""

    def __init__(self, documents, api_key, chunk_size=750, chunk_overlap=200):
        self.documents = documents
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retriever = self._build_retriever()

    def _split_documents(self):
        print("Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(self.documents)

    def _create_vector_store(self, doc_splits):
        print("Creating vector store...")
        embedding = OpenAIEmbeddings(openai_api_key=self.api_key)
        vectorstore = SKLearnVectorStore.from_documents(doc_splits, embedding=embedding)
        return vectorstore.as_retriever(k=4)

    def _build_retriever(self):
        doc_splits = self._split_documents()
        return self._create_vector_store(doc_splits)

    def get_retriever(self):
        return self.retriever
