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
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Create metadata for the document
            metadata = {
                "file_name": file_path
            }
            
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
            with open(file_path, 'r') as f:
                data = json.load(f)
            print(data)

            all_data = data.get('data', {})
            for course in all_data.get('classes', []):
                course_title = course.get('titleShort', 'No title')
                for group in course.get('enrollGroups', []):
                    for section in group.get('classSections', []):
                        # Extract meeting times, instructors, and notes for each section
                        section_info = []
                        
                        # Extracting meeting details
                        for meeting in section.get('meetings', []):
                            start_time = meeting.get('timeStart', 'N/A')
                            end_time = meeting.get('timeEnd', 'N/A')
                            days = meeting.get('pattern', 'N/A')
                            facility = meeting.get('facilityDescr', 'N/A')
                            section_info.append(f"Meets {days} from {start_time} to {end_time} at {facility}.")
                        
                        # Extracting instructor details
                        instructors = [
                            f"{instructor.get('firstName', '')} {instructor.get('lastName', '')}"
                            for instructor in meeting.get('instructors', [])
                        ]
                        if instructors:
                            section_info.append(f"Instructors: {', '.join(instructors)}")
                        
                        # Extracting class notes
                        notes = [
                            note.get('descrlong', 'No notes available')
                            for note in section.get('notes', [])
                        ]
                        if notes:
                            section_info.append(f"Notes: {' '.join(notes)}")
                        
                        # Create document content and metadata
                        content = f"Course: {course_title}\n" + "\n".join(section_info)
                        metadata = {
                            "course_id": course.get('crseId', 'N/A'),
                            "section": section.get('section', 'N/A'),
                            "class_number": section.get('classNbr', 'N/A'),
                            "instructors": instructors,
                            "notes": notes,
                        }
                        
                        # Append the document
                        docs.append(Document(page_content=content, metadata=metadata))

        print(f"Loaded {len(docs)} documents from {len(self.file_paths)} files.")
        # print(docs)
        return docs