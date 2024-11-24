from config.settings import Settings
from auth.auth_manager import AuthManager
from models.document_loader import TXTDocumentLoader
from models.retriever import Retriever
from controllers.rag_controller import RAGController
from views.console_view import ConsoleView

def main():
    try:
        # Load environment and settings
        Settings.load_environment()
        api_key = AuthManager.get_api_key()
        huggingface_model = "thenlper/gte-small"

        # Load course documents
        course_txts = [
            "data/data.txt"
        ]
        course_document_loader = TXTDocumentLoader(course_txts)
        course_documents = course_document_loader.load_documents()

        # Load graduation requirement documents
        grad_req_txts = [
            "data/gradRequirement.txt"
        ]
        grad_document_loader = TXTDocumentLoader(grad_req_txts)
        grad_documents = grad_document_loader.load_documents()

        # Initialize the retrievers
        course_retriever = Retriever(course_documents, huggingface_model, force_recompute=False).get_retriever()
        grad_retriever = Retriever(grad_documents, huggingface_model, force_recompute=False, top_k=1).get_retriever()

        # Example questions
        questions = ["Recommend me some courses about AI at Cornell"]

        # Initialize the RAG controller with both retrievers and questions
        rag_controller = RAGController(course_retriever, grad_retriever, questions)

        # Start the RAG application (handles both initial and user questions)
        rag_controller.run()

    except Exception as e:
        ConsoleView.display_error(e)

if __name__ == "__main__":
    main()
