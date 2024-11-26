from typing import List, Optional
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from views.console_view import ConsoleView  # Import the view for display


class RAGController:
    """Manages the Retrieval-Augmented Generation (RAG) process."""

    def __init__(
        self,
        course_retriever,
        grad_retriever,
        questions: Optional[List[str]] = None,
        is_debug=False,
    ):
        """
        Initializes the RAGController with an optional list of questions.

        Args:
            course_retriever: The retriever for course documents.
            grad_retriever: The retriever for graduation requirement documents.
            questions (Optional[List[str]]): A list of questions to be answered.
        """
        self.course_retriever = course_retriever
        self.grad_retriever = grad_retriever
        self.questions = questions or [] 
        self.grad_inference_chain = self._create_grad_inference_chain()
        self.answer_chain = self._create_answer_chain()
        self.is_debug = is_debug

    def _create_grad_inference_chain(self):
        """Creates the LLM chain for generating course description keywords from the question."""
        prompt = PromptTemplate(
            template="""
                You are an assistant that helps generate course description keywords from student queries.
                Given the student's question and relevant graduation requirement documents, generate a list of keywords or phrases that describe the type of courses that fufills the inferred graduation requirement.
                Focus on terms that will aid in retrieving relevant courses from the course embeddings.
                Make sure to follow the guidance from your retrieved document, give special attention to distribution requirement and course levels.
                Be as accurate as possible.
                Keep the keywords concise and relevant.
                Respond with only the list of keywords, NOTHING ELSE!!!

                Question: {question}
                Graduation Requirement Documents: {grad_docs}
                Course Description Keywords:
                """,
            input_variables=["question", "grad_docs"],
        )
        llm = ChatOllama(model="llama3.2", temperature=0)
        return prompt | llm | StrOutputParser()

    def _create_answer_chain(self):
        """Creates the LLM chain for generating the final answer."""
        prompt = PromptTemplate(
            template="""
            You are a professional academic advisor at Cornell University.
            Use the following course documents to answer the student's question.
            You can recommend courses that satisfy the inferred graduation requirements.
            Courses with lower course numbers are generally more entry-level.
            Prefer the lower level course when available.
            If you don't know the answer, just say that you don't know.
            Use 4 sentences MAXIMUM and keep the answer concise.

            Question: {question}
            Inferred Graduation Requirements: {inferred_grad_req}
            Course Documents: {course_docs}
            Answer:
            """,
            input_variables=["question", "inferred_grad_req", "course_docs"],
        )
        llm = ChatOllama(model="llama3.2", temperature=0)
        return prompt | llm | StrOutputParser()

    def answer_question(self, question: str) -> str:
        """Answers a single question and returns the answer."""
        if self.is_debug:
            print(f"QUESTION: {question}")

        grad_documents = self.grad_retriever.invoke(question)
        grad_doc_texts = "\n".join([doc.page_content for doc in grad_documents])

        # Infer graduation requirements using the grad inference chain
        inferred_grad_req = self.grad_inference_chain.invoke(
            {"question": question, "grad_docs": grad_doc_texts}
        ).strip()

        new_query = f"{question} {inferred_grad_req}"
        if self.is_debug:
            print("\n\n\n\nQuestion with Inferred Graduation Requirements:", new_query)

        # Retrieve course documents using the new query
        course_documents = self.course_retriever.invoke(new_query)
        course_doc_texts = "\n".join([doc.page_content for doc in course_documents])

        if self.is_debug:
            print("\n\n\n\nRetrieved Course Documents:", course_doc_texts)

        # Generate final response
        answer = self.answer_chain.invoke(
            {
                "question": question,
                "inferred_grad_req": inferred_grad_req,
                "course_docs": course_doc_texts,
            }
        ).strip()

        ConsoleView.display_message(f"\nANSWER: {answer}")
        ConsoleView.display_message("=========================================\n\n")
        return answer

    def interactive_loop(self):
        """Starts an interactive loop to answer initial and user-provided questions."""

        while self.questions:
            question = self.questions.pop(0) 
            self.answer_question(question)

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
