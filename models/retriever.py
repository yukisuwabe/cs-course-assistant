import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm


class Retriever:
    """Manages document processing and vector store creation with embedding caching."""

    def __init__(
        self,
        documents,
        model_path,
        chunk_size=400,
        chunk_overlap=10,
        data_folder="./data",
        force_recompute=False,
    ):
        """
        Initialize the Retriever.

        Args:
            documents: List of documents to process.
            model_path: Path to the local Hugging Face model directory.
            chunk_size: Maximum size of chunks in tokens.
            chunk_overlap: Overlap size between chunks in tokens.
            data_folder: Path to the folder where embeddings and vector stores are saved.
            force_recompute: If True, regenerate embeddings even if saved embeddings exist.
        """
        self.documents = documents
        self.model_path = model_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.data_folder = data_folder
        self.force_recompute = force_recompute
        self.retriever = self._build_retriever()

    def _split_documents(self):
        """Split documents into chunks."""
        print("Splitting documents into chunks...")
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
        embedding_file = os.path.join(self.data_folder, "embeddings.pkl")
        embeddings = []

        if os.path.exists(embedding_file) and not self.force_recompute:
            print(f"Loading existing embeddings from: {embedding_file}")
            with open(embedding_file, "rb") as f:
                embeddings = pickle.load(f)
        else:
            print("Generating new embeddings...")
            os.makedirs(self.data_folder, exist_ok=True)

            # Initialize the embedding model
            embedding = HuggingFaceEmbeddings(
                model_name=self.model_path,
                multi_process=False,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            # Progress tracking for embedding creation
            for doc in tqdm(doc_splits, desc="Embedding Documents", unit="doc"):
                embeddings.append(embedding.embed_documents([doc.page_content])[0])

            # Save the embeddings
            with open(embedding_file, "wb") as f:
                pickle.dump(embeddings, f)
            print(f"Saved embeddings to: {embedding_file}")

        return embeddings

    def _create_vector_store(self, doc_splits):
        """Create a vector store with local embeddings and progress tracking."""
        print("Creating vector store with FAISS...")

        # Load or generate embeddings
        embeddings = self._load_or_generate_embeddings(doc_splits)

        # Ensure embeddings are in NumPy format
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Prepare text_embedding_pairs
        text_embedding_pairs = [
            (doc.page_content, embedding)  # Tuple of text and corresponding embedding
            for doc, embedding in zip(doc_splits, embeddings_array)
        ]

        # Build FAISS vector store
        vectorstore = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=HuggingFaceEmbeddings(
                model_name=self.model_path,
                multi_process=False,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            ),
        )

        print("Vector store creation complete.")
        return vectorstore.as_retriever(k=4)

    def _build_retriever(self):
        """Build the retriever."""
        doc_splits = self._split_documents()
        return self._create_vector_store(doc_splits)

    def get_retriever(self):
        """Get the retriever."""
        return self.retriever
