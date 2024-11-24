from langchain_community.document_loaders import WebBaseLoader
import json
from langchain.schema import Document


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


class TXTDocumentLoader(DocumentLoaderBase):
    """Loads documents from multiple .txt files."""

    def __init__(self, file_paths):
        self.file_paths = file_paths

    def load_documents(self):
        docs = []

        # Load documents from each .txt file in the provided list of paths
        for file_path in self.file_paths:
            print(f"Loading document from {file_path}...")

            with open(file_path, "r") as f:
                content = f.read()

            # Create metadata for the document
            metadata = {"file_name": file_path}

            # Append the document
            docs.append(Document(page_content=content, metadata=metadata))

        print(f"Loaded {len(docs)} documents from {len(self.file_paths)} files.")
        return docs


class JSONDocumentLoader(DocumentLoaderBase):
    """Loads documents from multiple JSON files."""

    def __init__(self, file_paths):
        self.file_paths = file_paths

    def load_documents(self):
        docs = []

        # Load documents from each JSON file in the provided list of paths
        for file_path in self.file_paths:
            print(f"Loading documents from {file_path}...")
            with open(file_path, "r") as f:
                data = json.load(f)
            # print(data)

            all_data = data.get("classes", {})
            for subject in all_data:
                for course in all_data[subject]:
                    course_code = (
                        course.get("subject", "N/A")
                        + " "
                        + course.get("catalogNbr", "")
                    )
                    course_title = course.get("titleLong", "No title")
                    description = course.get("description", "No description")
                    outcomes = course.get("catalogOutcomes", [])
                    if not outcomes:
                        outcomes = []
                    distribution = course.get("catalogDistr", "N/A")
                    content = (
                        f"Course: {course_code}\n"
                        + f"Course Title: {course_title}\n"
                        + f"Description: {description}\n"
                        + f"Outcomes: {', '.join(outcomes)}\n"
                        + f"Distribution: {distribution}\n"
                    )

                    # Append the document
                    docs.append(Document(page_content=content))

        print(f"Loaded {len(docs)} documents from {len(self.file_paths)} files.")
        # print(docs)
        return docs
