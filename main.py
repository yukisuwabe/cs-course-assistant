from config.settings import Settings
from auth.auth_manager import AuthManager
from models.document_loader import TXTDocumentLoader
from models.retriever import Retriever
from controllers.rag_controller import RAGController
from views.console_view import ConsoleView

def main():
    try:
        # Load environment and settings
        # Settings.load_environment()
        # api_key = AuthManager.get_api_key()
        huggingface_model = "thenlper/gte-small"

        # LOAD DOCUMENTS
        course_txts = [
            "data/data.txt"
        ]
        course_document_loader = TXTDocumentLoader(course_txts)
        course_documents = course_document_loader.load_documents()

        grad_req_txts = [
            "data/gradRequirement.txt"
        ]
        grad_document_loader = TXTDocumentLoader(grad_req_txts)
        grad_documents = grad_document_loader.load_documents()

        # BUILD RETRIEVERS
        course_retriever = Retriever(
            documents=course_documents,
            model_path=huggingface_model,
            retriever_name="course", 
            force_recompute=False,
            top_k=8
        ).get_retriever()

        grad_retriever = Retriever(
            documents=grad_documents,
            model_path=huggingface_model,
            retriever_name="grad", 
            force_recompute=False,
            top_k=2
        ).get_retriever()

        # PREDEFINED QUESTIONS
        questions = ["Recommend me some courses about AI at Cornell"]
        ConsoleView.display_message("Default Questions", questions)
        rag_controller = RAGController(course_retriever, grad_retriever, questions, is_debug=False)

        rag_controller.run()

    except Exception as e:
        ConsoleView.display_error(e)

if __name__ == "__main__":
    main()
