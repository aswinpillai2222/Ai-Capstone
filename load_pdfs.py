from rag_tools import load_pdf_files, extract_text_from_pdf, split_text_in_chunks, vectorize_text, add_vectors_to_db

pdfs = load_pdf_files('./pdfs')

for i, pdf in enumerate(pdfs):
  print(f"Processing {i+1} of {len(pdfs)}: {pdf}")
  text = extract_text_from_pdf(pdf)
  text_chunks = split_text_in_chunks(text)
  vectors = vectorize_text(text_chunks)
  add_vectors_to_db(pdf,text_chunks, vectors)