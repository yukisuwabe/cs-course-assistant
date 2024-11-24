from typing import List, Optional
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from views.console_view import ConsoleView  # Import the view for display

class RAGController:
    """Manages the Retrieval-Augmented Generation (RAG) process."""

    def __init__(self, retriever, grad_retriever, questions: Optional[List[str]] = None):
        """
        Initializes the RAGController with an optional list of questions.

        Args:
            retriever: The retriever to use for course document retrieval.
            grad_retriever: The retriever to use for graduation requirement retrieval.
            questions (Optional[List[str]]): A list of questions to be answered.
        """
        self.retriever = retriever
        self.grad_retriever = grad_retriever  # New retriever for graduation requirements
        self.questions = questions or []  # Initialize with provided questions or an empty list
        self.rag_chain = self._create_rag_chain()

    def _create_rag_chain(self):
        """Creates the RAG chain combining the LLM and prompt template."""
        prompt = PromptTemplate(
            template="""
                You are now a professional academic advisor at Cornell University.
                Use the following documents to answer the question.
                You can recommend similar courses if the one specified is not found.
                Courses with lower course number are generally more entry level.
                If the user asks about course information, use the course description for reference.
                If you don't know the answer, just say that you don't know.
                If you are asked about degree progress, refer to the Graduation Requirements provided.
                Use 4 sentences MAXIMUM and keep the answer concise:

                Question: {question}
                Documents: {documents}
                Graduation Requirements: {gradreq}
                Answer:
                """,
            input_variables=["question", "documents", "gradreq"],
        )
        llm = ChatOllama(model="llama3.2", temperature=0)
        return prompt | llm | StrOutputParser()

    def answer_question(self, question: str) -> str:
        """Answers a single question and returns the answer."""
        ConsoleView.display_message(f"QUESTION: {question}")
        # Retrieve course documents
        documents = self.retriever.invoke(question)
        doc_texts = "\n".join([doc.page_content for doc in documents])
        # Retrieve graduation requirement documents
        grad_documents = self.grad_retriever.invoke(question)
        grad_doc_texts = "\n".join([doc.page_content for doc in grad_documents])

        # # Prepare inputs for the chain
        # chain_inputs = {
        #     "question": question,
        #     "documents": doc_texts,
        #     "gradreq": grad_doc_texts
        # }

        # # Generate the final prompt
        # final_prompt = self.rag_chain.prompt.format(**chain_inputs)
        # print("Final Prompt Sent to LLM:\n", final_prompt)

        # Pass both documents to the LLM
        answer = self.rag_chain.invoke({
            "question": question,
            "documents": doc_texts,
            "gradreq": grad_doc_texts
        })
        ConsoleView.display_message(f"\nANSWER: {answer}")
        ConsoleView.display_message("=========================================\n\n")
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
