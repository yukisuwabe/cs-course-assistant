from typing import List, Optional
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from views.console_view import ConsoleView  # Import the view for display

class RAGController:
    """Manages the Retrieval-Augmented Generation (RAG) process."""

    def __init__(self, retriever, questions: Optional[List[str]] = None):
        """
        Initializes the RAGController with an optional list of questions.
        
        Args:
            retriever: The retriever to use for document retrieval.
            questions (Optional[List[str]]): A list of questions to be answered.
        """
        self.retriever = retriever
        self.questions = questions or []  # Initialize with provided questions or an empty list
        self.rag_chain = self._create_rag_chain()

    def _create_rag_chain(self):
        """Creates the RAG chain combining the LLM and prompt template."""
        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks.
            Use the following documents to answer the question.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise:
            Question: {question}
            Documents: {documents}
            Answer:
            """,
            input_variables=["question", "documents"],
        )
        llm = ChatOllama(model="llama3.2", temperature=0)
        return prompt | llm | StrOutputParser()

    def answer_question(self, question: str) -> str:
        """Answers a single question and returns the answer."""
        ConsoleView.display_message(f"Question: {question}")
        documents = self.retriever.invoke(question)
        doc_texts = "\n".join([doc.page_content for doc in documents])
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        ConsoleView.display_message(f"Answer: {answer}")
        return answer

    def interactive_loop(self):
        """Starts an interactive loop to answer initial and user-provided questions."""
        # First, answer any initial questions
        while self.questions:
            question = self.questions.pop(0)  # Get the first question from the list
            self.answer_question(question)

        # Then, take user input until the user types ':q'
        while True:
            user_input = input("Enter your question (:q to quit): ").strip()
            if user_input.lower() == ":q":
                ConsoleView.display_message("Exiting. Goodbye!")
                break
            self.answer_question(user_input)

    def run(self):
        """Kicks off the RAG application by entering the interactive loop."""
        ConsoleView.display_message("Starting RAG Application...")
        self.interactive_loop()
