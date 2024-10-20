# ai-estein's disciples

We are building a knowledge-based chatbot using a fine-tuned Llama model that assists Cornell CS students with course selection by providing personalized recommendations based on course reviews, descriptions, and degree requirements.

Yuki Suwabe: Talk about data scraping and data cleanup. How the Course Roster API was utilized and cleaned up. Extracted data such as class code, title, description, class outcome if it exists, and distribution category if it exists to ensure data that matters gets entered and is easier for the data to be vectorized. Included class outcome for a general goal that the student may want to get out of a class and distribution category if the student is more concerned about finding classes that satisfy their requirements. Will demo the data extraction using the API using a small subset of classes and the cleanup of those codes.

Peter Wu: I will walk through how I set up the LLM on my local machine (so that people can reproduce my work), including topics about Llama 3.2 with Ollama, the Langchain module, the document vectorization process, the retrieval process, prompt engineering, and codebase setup. After details of the technicalities, I will talk about the overall plans and architecture of our project, including going through the flowchart and reporting on where we are currently. I can then talk about some immediate next steps, including possibly switching to Google Collab, proposals for finetuning, and exploring faster and more powerful libraries.
