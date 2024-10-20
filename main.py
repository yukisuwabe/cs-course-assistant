from config.settings import Settings
from auth.auth_manager import AuthManager
from models.document_loader import URLDocumentLoader, JSONDocumentLoader, TXTDocumentLoader
from models.retriever import Retriever
from controllers.rag_controller import RAGController
from views.console_view import ConsoleView

def main():
    try:
        # Load environment and settings
        Settings.load_environment()
        api_key = AuthManager.get_api_key()

        # Load documents from URLs
        urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]
        jsons = [
            "data/CSClasses.json"
        ]
        txts = [
            "data/data.txt"
        ]
        document_loader = TXTDocumentLoader(txts)
        documents = document_loader.load_documents()

        input("Press any key to continue...")

        # Initialize the retriever
        retriever = Retriever(documents, api_key).get_retriever()

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
