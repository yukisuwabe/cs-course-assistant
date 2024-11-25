import unittest
import pickle
from controllers.rag_controller import RAGController
from models.document_loader import (
    URLDocumentLoader,
    TXTDocumentLoader,
    JSONDocumentLoader,
)
from models.retriever import Retriever
from langchain.schema import Document
from unittest.mock import patch, mock_open, MagicMock, Mock
from config.settings import Settings
import numpy as np
import os
from types import SimpleNamespace
from views.console_view import ConsoleView


class TestDocumentLoaders(unittest.TestCase):

    @patch("models.document_loader.WebBaseLoader.load")
    def test_url_document_loader(self, mock_load):
        mock_load.return_value = [Document(page_content="Test content", metadata={})]
        urls = ["http://example.com/doc1", "http://example.com/doc2"]
        loader = URLDocumentLoader(urls)
        documents = loader.load_documents()
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].page_content, "Test content")

    # @patch("builtins.open", new_callable=mock_open, read_data="Test content")
    def test_txt_document_loader(self):
        file_paths = ["./data/gradRequirement.txt"]
        loader = TXTDocumentLoader(file_paths)
        documents = loader.load_documents()

        with open("./data/gradRequirement.txt", "r") as f:
            content = f.read()

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, content)
        self.assertEqual(
            documents[0].metadata["file_name"], "./data/gradRequirement.txt"
        )

    def test_json_document_loader(self):
        file_paths = ["./data/CSClasses.json", "./data/INFOClasses.json"]
        loader = JSONDocumentLoader(file_paths)
        documents = loader.load_documents()
        self.assertIn("Course: CS 1110", documents[0].page_content)
        self.assertIn(
            "Course Title: Introduction to Computing: A Design and Development Perspective",
            documents[0].page_content,
        )
        self.assertIn(
            "Description: Programming and problem solving using Python. Emphasizes principles of software development, style, and testing. Topics include procedures and functions, iteration, recursion, arrays and vectors, strings, an operational model of procedure and function calls, algorithms, exceptions, object-oriented programming. Weekly labs provide guided practice on the computer, with staff present to help. ",
            documents[0].page_content,
        )
        self.assertIn(
            "Outcomes: Be fluent in the use of procedural statements -assignments, conditional statements, loops, method calls- and  arrays., Be able to design, code, and test small Python programs that meet requirements expressed in English. This includes a basic understanding of top-down design., Understand the concepts of object-oriented programming as used in Python: classes, subclasses, inheritance, and overriding., Have knowledge of basic searching and sorting algorithms. Have knowledge of the basics of vector computation.",
            documents[0].page_content,
        )
        self.assertIn("Distribution: (SMR-AS)", documents[0].page_content)


class TestRetrievers(unittest.TestCase):
    @patch("models.retriever.Retriever._build_retriever", return_value=None)
    def test_retriever_initialization(self, mock_build_retriever):
        documents = [{"text": "This is a sample document for testing."}]
        retriever = Retriever(
            documents=documents,
            model_path="path/to/model",
            chunk_size=100,
            chunk_overlap=10,
            data_folder="./test_data",
            force_recompute=True,
        )
        self.assertIsInstance(retriever, Retriever)
        self.assertEqual(retriever.documents, documents)
        self.assertIn(retriever.model_path, "path/to/model")
        self.assertEqual(retriever.chunk_size, 100)
        self.assertEqual(retriever.chunk_overlap, 10)
        self.assertEqual(retriever.data_folder, "./test_data")
        self.assertTrue(retriever.force_recompute)

    @patch("models.retriever.Retriever._build_retriever", return_value=None)
    @patch("models.retriever.AutoTokenizer.from_pretrained", return_value=1)
    @patch(
        "models.retriever.RecursiveCharacterTextSplitter.from_huggingface_tokenizer",
        return_value=MagicMock(),
    )
    def test_split_documents(
        self,
        mock_splitter,
        mock_tokenizer,
        mock_build_retriever,
    ):
        mock_splitter_instance = MagicMock()
        mock_splitter.return_value = mock_splitter_instance

        mock_splitter_instance.split_documents = MagicMock(
            return_value=[
                {"page_content": "Chunk 1"},
                {"page_content": "Chunk 2"},
            ]
        )
        documents = [Document(page_content="Test content", metadata={})]
        retriever = Retriever(documents, "path/to/model")

        chunks = retriever._split_documents()

        mock_tokenizer.assert_called_once_with("path/to/model")
        mock_splitter.assert_called_once()
        mock_splitter_instance.split_documents.assert_called_once_with(documents)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["page_content"], "Chunk 1")
        self.assertEqual(chunks[1]["page_content"], "Chunk 2")

    @patch("models.retriever.Retriever._build_retriever", return_value=None)
    @patch(
        "models.retriever.Retriever._split_documents",
    )
    @patch("models.retriever.HuggingFaceEmbeddings")
    def test_load_or_generate_embeddings_new(
        self, mock_embedding, mock_split_documents, mock_build_retriever
    ):
        documents = [{"text": "This is a sample document for embedding."}]
        retriever = Retriever(
            documents=documents,
            model_path="path/to/model",
            data_folder="./test_data",
            force_recompute=True,
        )
        mock_split_documents.return_value = [
            SimpleNamespace(**{"page_content": "Mock Chunk 1"}),
            SimpleNamespace(**{"page_content": "Mock Chunk 2"}),
        ]
        doc_splits = retriever._split_documents()

        mock_embedding.return_value.embed_documents.return_value = [np.array([1, 2, 3])]

        embeddings = retriever._load_or_generate_embeddings(doc_splits)

        self.assertEqual(len(embeddings), len(doc_splits))
        self.assertIsInstance(embeddings, list)
        self.assertTrue(os.path.exists("./test_data/embeddings.pkl"))

    @patch("models.retriever.Retriever._build_retriever", return_value=None)
    @patch(
        "models.retriever.Retriever._split_documents",
    )
    @patch("models.retriever.HuggingFaceEmbeddings")
    @patch("models.retriever.pickle.load", wraps=pickle.load)
    def test_load_or_generate_embeddings_exists(
        self,
        mock_pickle_load,
        mock_embedding,
        mock_split_documents,
        mock_build_retriever,
    ):
        documents = [{"text": "This is a sample document for embedding."}]
        retriever = Retriever(
            documents=documents,
            model_path="path/to/model",
            data_folder="./test_data",
            force_recompute=False,
        )
        mock_split_documents.return_value = [
            SimpleNamespace(**{"page_content": "Mock Chunk 1"}),
            SimpleNamespace(**{"page_content": "Mock Chunk 2"}),
        ]
        doc_splits = retriever._split_documents()

        mock_embedding.return_value.embed_documents.return_value = [np.array([1, 2, 3])]

        embeddings = retriever._load_or_generate_embeddings(doc_splits)

        mock_pickle_load.assert_called_once()
        self.assertEqual(len(embeddings), len(doc_splits))
        self.assertIsInstance(embeddings, list)
        self.assertTrue(os.path.exists("./test_data/embeddings.pkl"))

    @patch("models.retriever.Retriever._build_retriever", return_value=None)
    @patch(
        "models.retriever.Retriever._split_documents",
    )
    @patch("models.retriever.HuggingFaceEmbeddings")
    @patch("models.retriever.FAISS")
    def test_create_vector_store(
        self, mock_faiss, mock_embedding, mock_split_documents, mock_build_retriever
    ):
        documents = [{"text": "This is a test document for vector store creation."}]
        retriever = Retriever(
            documents=documents,
            model_path="path/to/model",
            data_folder="./test_data",
            force_recompute=True,
        )
        mock_split_documents.return_value = [
            SimpleNamespace(**{"page_content": "Mock Chunk 1"}),
            SimpleNamespace(**{"page_content": "Mock Chunk 2"}),
        ]
        doc_splits = retriever._split_documents()

        mock_embedding.return_value.embed_documents.return_value = [np.array([1, 2, 3])]
        mock_faiss.from_embeddings.return_value.as_retriever.return_value = (
            "mock_retriever"
        )

        retriever_obj = retriever._create_vector_store(doc_splits)
        self.assertEqual(retriever_obj, "mock_retriever")

    @patch("models.retriever.AutoTokenizer.from_pretrained")
    @patch("models.retriever.RecursiveCharacterTextSplitter")
    @patch("models.retriever.HuggingFaceEmbeddings")
    @patch("models.retriever.FAISS")
    def test_retriever_building(
        self, mock_faiss, mock_embedding, mock_splitter, mock_tokenizer
    ):
        documents = [{"text": "End-to-end testing of the retriever pipeline."}]
        retriever = Retriever(
            documents=documents,
            model_path="path/to/model",
            data_folder="./test_data",
            force_recompute=True,
        )

        retriever_obj = retriever.get_retriever()
        self.assertNotEqual(retriever_obj, None)


class TestRAG(unittest.TestCase):
    @patch("controllers.rag_controller.RAGController.run")
    @patch("controllers.rag_controller.RAGController.__init__")
    def test_rag_controller(self, mock_init, mock_run):
        mock_init.return_value = None
        rag_controller = RAGController(Mock(), ["Test question"])
        rag_controller.run()
        mock_run.assert_called_once()

    def test_create_rag_chain(self):
        retriever = Mock()
        rag_controller = RAGController(retriever)
        rag_chain = rag_controller._create_rag_chain()
        self.assertIsNotNone(rag_chain)

    def test_rag_answer_question(self):
        Settings.load_environment()
        huggingface_model = "thenlper/gte-small"

        txts = ["data/data.txt", "data/gradRequirement.txt"]
        document_loader = TXTDocumentLoader(txts)
        documents = document_loader.load_documents()

        retriever = Retriever(documents, huggingface_model).get_retriever()

        rag_controller = RAGController(retriever)

        answer = rag_controller.answer_question(
            "What is a good course to learn Object Oriented Programming?"
        )
        self.assertIn("CS 2110", answer)

    @patch("builtins.input", side_effect=["What is AI?", ":q"])
    @patch.object(ConsoleView, "display_message")
    @patch.object(RAGController, "answer_question")
    def test_interactive_loop(
        self, mock_answer_question, mock_display_message, mock_input
    ):
        retriever = MagicMock()
        questions = ["What is a good class to take to learn Python?"]
        rag_controller = RAGController(retriever, questions)

        rag_controller.interactive_loop()

        mock_answer_question.assert_any_call(
            "What is a good class to take to learn Python?"
        )
        mock_answer_question.assert_any_call("What is AI?")
        mock_display_message.assert_any_call("Exiting. Goodbye!")


if __name__ == "__main__":
    unittest.main()
