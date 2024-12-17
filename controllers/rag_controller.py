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
        """Creates the LLM chain for generating both concise course keywords and clear requirement info from the question."""
        prompt = PromptTemplate(
            template="""
                You are an assistant that helps generate course description keywords and minimal requirement details from student queries.
                Given the student's question and the relevant graduation requirement documents, generate a concise list (up to 6 short sentences) that includes:
                1) Keywords or phrases describing the types of courses that fulfill the inferred graduation requirement.
                2) A brief description of the actual graduation requirement itself and any important conditions (e.g., required course level, distribution category, department, or prerequisite details).
                3) DO NOT Hallucinate anything if it is not in the document, just say no specific requirement needed.

                Be sure to say that grad requirement is not important when it is not!

                Focus on terms that will aid in retrieving and selecting the relevant courses from the embeddings.
                Be accurate, concise, and follow the guidance from the provided documents.
                Respond with only the list of short sentences, NOTHING ELSE!!!
                
                Question: {question}
                Graduation Requirement Documents: {grad_docs}
                Course Description Keywords and Requirement Info:
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
            Provide both lower, middle, and higher ranged course recommendation when the level varies.
            Output in natural languages as an advisor, not in bullet points.
            Focus on the course that is most related to the question if all satisfy the grad requirement.
            If you don't know the answer, just say that you don't know.
            Make sure to consider prompts from Infeered Graduation Requirements carefully.
            Use 6 sentences MAXIMUM and keep the answer concise.

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
