from config.settings import Settings
from auth.auth_manager import AuthManager
from models.document_loader import URLDocumentLoader, JSONDocumentLoader, TXTDocumentLoader
from models.retriever import Retriever
from controllers.rag_controller import RAGController
from views.console_view import ConsoleView

GENERATE_NEW_EMBEDDING = False

def main():
    try:
        # Load environment and settings
        Settings.load_environment()
        api_key = AuthManager.get_api_key()
        huggingface_model = "thenlper/gte-small"

        if GENERATE_NEW_EMBEDDING:
            # Load documents from URLs
            txts = [
                "data/data.txt"
            ]
            document_loader = TXTDocumentLoader(txts)
            documents = document_loader.load_documents()

        # Initialize the retriever
        retriever = Retriever(documents, huggingface_model, force_recompute=GENERATE_NEW_EMBEDDING).get_retriever()

        # Example questions
        questions = ["Recommend me some course about AI at Cornell"]

        # Initialize the RAG controller with questions
        rag_controller = RAGController(retriever, questions)

        # Start the RAG application (handles both initial and user questions)
        rag_controller.run()

    except Exception as e:
        ConsoleView.display_error(e)

if __name__ == "__main__":
    main()
