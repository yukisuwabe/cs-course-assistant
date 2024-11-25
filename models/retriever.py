import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

class Retriever:
    """Manages document processing and vector store creation with embedding caching."""

    def __init__(
        self,
        documents,
        model_path,
        retriever_name="default",
        chunk_size=200,
        chunk_overlap=10,
        data_folder="./data",
        force_recompute=False,
        top_k=15,
    ):
        """
        Initialize the Retriever.

        Args:
            documents: List of documents to process.
            model_path: Path to the local Hugging Face model directory.
            retriever_name: Unique name for the retriever to prevent file overwrites.
            chunk_size: Maximum size of chunks in tokens.
            chunk_overlap: Overlap size between chunks in tokens.
            data_folder: Path to the folder where embeddings and vector stores are saved.
            force_recompute: If True, regenerate embeddings even if saved embeddings exist.
            top_k: Number of top results to retrieve.
        """
        self.documents = documents
        self.model_path = model_path
        self.retriever_name = retriever_name  # Unique name for the retriever
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.data_folder = data_folder
        self.force_recompute = force_recompute
        self.top_k = top_k
        self.retriever = self._build_retriever()

    def _split_documents(self):
        """Split documents into chunks."""
        print(f"Splitting documents into chunks for retriever '{self.retriever_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
        )
        return splitter.split_documents(self.documents)

    def _load_or_generate_embeddings(self, doc_splits):
        """Load embeddings from file or generate them if not available or forced to recompute."""
        embeddings_file = os.path.join(self.data_folder, f"embeddings_{self.retriever_name}.pkl")
        embeddings = []

        if os.path.exists(embeddings_file) and not self.force_recompute:
            print(f"Loading existing embeddings from: {embeddings_file}")
            with open(embeddings_file, "rb") as f:
                embeddings = pickle.load(f)
        else:
            print(f"Generating new embeddings for retriever '{self.retriever_name}'...")
            os.makedirs(self.data_folder, exist_ok=True)

            embedding_model = HuggingFaceEmbeddings(
                model_name=self.model_path,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            # Progress tracking for embedding creation
            for doc in tqdm(doc_splits, desc="Embedding Documents", unit="doc"):
                embeddings.append(embedding_model.embed_documents([doc.page_content])[0])

            with open(embeddings_file, "wb") as f:
                pickle.dump(embeddings, f)
            print(f"Saved embeddings to: {embeddings_file}")
        
        return embeddings

    def _create_vector_store(self, doc_splits):
        """Create a vector store with local embeddings and progress tracking."""
        print(f"Creating vector store with FAISS for retriever '{self.retriever_name}'...")

        embeddings = self._load_or_generate_embeddings(doc_splits)
        embeddings_array = np.array(embeddings, dtype=np.float32)

        text_embedding_pairs = [
            (doc.page_content, embedding)  # Tuple of text and corresponding embedding
            for doc, embedding in zip(doc_splits, embeddings_array)
        ]

        vectorstore = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=HuggingFaceEmbeddings(
                model_name=self.model_path,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            ),
        )
        print(f"Vector store creation complete for retriever '{self.retriever_name}'.")
        return vectorstore.as_retriever(search_kwargs={'k': self.top_k})

    def _build_retriever(self):
        """Build the retriever."""
        doc_splits = self._split_documents()
        return self._create_vector_store(doc_splits)

    def get_retriever(self):
        """Get the retriever."""
        return self.retriever
