from langchain_community.document_loaders import WebBaseLoader

class DocumentLoaderBase:
    """Abstract base class for loading documents."""
    def load_documents(self):
        """
        Loads documents from a specific source.

        Returns:
            List[Document]: A list of Document objects, each containing content 
                            and metadata.
        """

        raise NotImplementedError("This method must be implemented by subclasses.")

class URLDocumentLoader(DocumentLoaderBase):
    """Loads documents from a list of URLs."""
    def __init__(self, urls):
        self.urls = urls

    def load_documents(self):
        print("Loading documents from URLs...")
        docs = [WebBaseLoader(url).load() for url in self.urls]
        docs_list = [item for sublist in docs for item in sublist]
        print(f"Loaded {len(docs_list)} documents.")
        print(docs_list)
        return docs_list
