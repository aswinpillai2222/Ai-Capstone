from model import ask_llm
from rag_tools import query_vector_db

def ask_rag(prompt):
  # Query the vector database
  chunks, related_files = query_vector_db(prompt)

  if len(chunks) == 0:
    print("No context found for the prompt")
    return "Sorry, I don't have any information about that topic.", []

  context = "\n".join(chunks)

  # Augment the prompt with the context
  input_text = (
      f"Use the following pieces of context to provide a concise and straight answer to the question. "
      f"Do not repeat the context verbatim. Focus on summarizing the key points and providing a clear response.\n\n"
      f"Context: {context}\n\n"
      f"Question: {prompt}\n\n"
      f"Helpful Answer:"
  )

  # Ask the LLM
  response = ask_llm(input_text)

  return response, related_files
