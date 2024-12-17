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

# Change here to specify which test class you want to run
TESTS_TO_RUN = ['TestAnswers']

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
        if os.path.exists("./test_data/embeddings_default.pkl"):
            os.remove("./test_data/embeddings_default.pkl")
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
        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), len(doc_splits))
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
            force_recompute=False,
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
            force_recompute=False,
        )

        retriever_obj = retriever.get_retriever()
        self.assertNotEqual(retriever_obj, None)


class TestRAG(unittest.TestCase):
    @patch("controllers.rag_controller.RAGController._create_grad_inference_chain")
    @patch("controllers.rag_controller.RAGController._create_answer_chain")
    def test_rag_controller_init(self, mock_answer_chain, mock_grad_chain):
        rag_controller = RAGController(
            "course", "grad", ["question1", "question2"], is_debug=False
        )
        self.assertEqual(rag_controller.course_retriever, "course")
        self.assertEqual(rag_controller.grad_retriever, "grad")
        self.assertEqual(rag_controller.questions, ["question1", "question2"])
        mock_grad_chain.assert_called_once()
        mock_answer_chain.assert_called_once()
        self.assertFalse(rag_controller.is_debug)

    def test_create_grad_inference_chain(self):
        course_retriever = Mock()
        grad_retriever = Mock()
        rag_controller = RAGController(course_retriever, grad_retriever)
        rag_chain = rag_controller._create_grad_inference_chain()
        self.assertIsNotNone(rag_chain)

    def test_create_answer_chain(self):
        course_retriever = Mock()
        grad_retriever = Mock()
        rag_controller = RAGController(course_retriever, grad_retriever)
        rag_chain = rag_controller._create_answer_chain()
        self.assertIsNotNone(rag_chain)

    @patch("builtins.input", side_effect=["What is AI?", ":q"])
    @patch.object(ConsoleView, "display_message")
    @patch.object(RAGController, "answer_question")
    def test_interactive_loop(
        self, mock_answer_question, mock_display_message, mock_input
    ):
        course_retriever = MagicMock()
        grad_retriever = MagicMock()
        questions = ["What is a good class to take to learn Python?"]
        rag_controller = RAGController(
            course_retriever, grad_retriever, questions, is_debug=False
        )

        rag_controller.interactive_loop()

        mock_answer_question.assert_any_call(
            "What is a good class to take to learn Python?"
        )
        mock_answer_question.assert_any_call("What is AI?")
        mock_display_message.assert_any_call("Exiting. Goodbye!")


class TestAnswers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Settings.load_environment()
        huggingface_model = "thenlper/gte-small"

        course_txts = ["data/data.txt"]
        course_document_loader = TXTDocumentLoader(course_txts)
        course_documents = course_document_loader.load_documents()

        grad_req_txts = ["data/gradRequirement.txt"]
        grad_document_loader = TXTDocumentLoader(grad_req_txts)
        grad_documents = grad_document_loader.load_documents()

        course_retriever = Retriever(
            documents=course_documents,
            model_path=huggingface_model,
            retriever_name="course",
            force_recompute=False,
            top_k=8,
        ).get_retriever()

        grad_retriever = Retriever(
            documents=grad_documents,
            model_path=huggingface_model,
            retriever_name="grad",
            force_recompute=False,
            top_k=2,
        ).get_retriever()

        cls.rag_controller = RAGController(
            course_retriever, grad_retriever, is_debug=True
        )

    def test_rag_answer_question_2110(self):
        answer = self.rag_controller.answer_question(
            "What is a good course to learn Object Oriented Programming?"
        )
        self.assertIn("CS 2110", answer)

    def test_rag_answer_question_3780(self):
        answer = self.rag_controller.answer_question(
            "What are some introductory machine learning classes you would recommend?"
        )
        self.assertIn("CS 3780", answer)

    def test_rag_answer_question_4620(self):
        answer = self.rag_controller.answer_question(
            "Is there any 3000 or 4000 technical electives that explores how computers are used to generate graphics?"
        )
        self.assertIn("CS 4620", answer)

    def test_rag_answer_question_4820(self):
        answer = self.rag_controller.answer_question(
            "I want to learn more about algorithms and data structures, what are some required classes for CS major that explores these fields?"
        )
        self.assertIn("CS 4820", answer)

    def test_rag_answer_question_4701(self):
        answer = self.rag_controller.answer_question(
            "After learning some basics about AI, I want to put them into practice. What is the best practicum class to do so?"
        )
        self.assertIn("CS 4701", answer)

    def test_rag_answer_question_3300(self):
        answer = self.rag_controller.answer_question(
            "What class do you know that teaches how to visualize data with a webpage?"
        )
        self.assertIn("3300", answer)

    def test_rag_answer_question_4750_4756(self):
        answer = self.rag_controller.answer_question(
            "Do you know of any computer science classes that are related to robot building and software?"
        )
        self.assertTrue("CS 4750" in answer or "CS 4756" in answer)

    def test_rag_answer_question_3410(self):
        answer = self.rag_controller.answer_question(
            "Which Cornell required CS core course teaches you about the architecture of computers?"
        )
        self.assertIn("CS 3410", answer)

    def test_rag_answer_question_4320(self):
        answer = self.rag_controller.answer_question(
            "I want to learn more about how large groups of data are stored and learn how databases work, which course should I take?"
        )
        self.assertIn("CS 4320", answer)

    def test_rag_answer_question_as_requirement(self):
        answer = self.rag_controller.answer_question(
            "I am a Arts and Sciences student. If I took AMST 2006, which distribution requirement would it fulfill?"
        )
        self.assertIn("ALC-AS", answer)

    def test_rag_answer_question_eng_requirement(self):
        answer = self.rag_controller.answer_question(
            "I am a College of Engineering student. If I already took MATH 1910 and MATH 1920, which math classes do I need to take to satisfy my math requirements?"
        )
        self.assertTrue("2930" in answer or "2940" in answer)

    @classmethod
    def tearDownClass(cls):
        del cls.rag_controller


def run_selected_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for test in TESTS_TO_RUN:
        if "." in test:
            suite.addTest(loader.loadTestsFromName(test))  # Specific test method
        else:
            suite.addTests(loader.loadTestsFromTestCase(globals()[test]))  # Test class

    # Redirect output to test_result.txt
    with open("test_result.txt", "w") as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        print("Running selected tests. Results are saved in test_result.txt")
        runner.run(suite)


if __name__ == "__main__":
    run_selected_tests()

