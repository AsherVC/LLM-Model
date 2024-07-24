import nltk
from tqdm import tqdm
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np

# Download punkt if not already downloaded
nltk.download('punkt')

def extract_text_from_pdfs(pdf_paths):
    texts = []
    for pdf_path in pdf_paths:
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            for page_num in tqdm(range(num_pages), desc=f"Extracting {pdf_path}"):
                page = reader.pages[page_num]
                text += page.extract_text()
        texts.append(text)
    return texts

def chunk_text(text, chunk_size=100):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.extend(nltk.word_tokenize(sentence))
        if len(current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def write_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

# Paths to your PDF files
pdf_paths = [
    "C:/Users/M S I/steps/alppaydin_machinelearning_2010.pdf",
    "C:/Users/M S I/steps/Francis X. Govers - Artificial Intelligence for Robotics-Packt.pdf",
    "C:/Users/M S I/steps/FUNDAMENTALS OF NEURAL NETWORKS_LAURENE FAUSETT.pdf"
]

# Extract text from the PDFs
textbook_texts = extract_text_from_pdfs(pdf_paths)

# Chunk each extracted text and save the chunks to text files
all_chunks = []
for i, text in enumerate(textbook_texts):
    chunks = chunk_text(text)
    all_chunks.extend(chunks)
    for j, chunk in enumerate(chunks):
        file_path = f"textbook_{i+1}_chunk_{j+1}.txt"
        write_text_to_file(chunk, file_path)
        print(f"Chunk {j+1} of textbook {i+1} written to {file_path}")

# Initialize the Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Embed each chunk
embeddings = model.encode(all_chunks)

# Save the embeddings and chunks for the next steps
np.save("embeddings.npy", embeddings)
with open("chunks.txt", "w", encoding="utf-8") as f:
    for chunk in all_chunks:
        f.write(chunk + "\n")

# Save the model (if needed)
model.save("sentence_transformer_model")
