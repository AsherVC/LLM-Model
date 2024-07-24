This project is a take-home assessment for an NLP Intern position at Steps AI. The objective is to create a question-answering system based on the content of three textbooks.

## Project Structure

- `extract_and_chunk.py`: Script to extract content from PDFs and chunk the text.
- `embedding_and_cluster.py`: Script to embed the chunks using SBERT and cluster them using Gaussian Mixture Models.
- `query_retrieval.py`: Script to insert embeddings into Milvus and perform query retrieval.
- `requirements.txt`: List of required Python packages.
- `data/`: Directory containing the extracted text chunks and embeddings.

## Setup and Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-username/your-repository-name.git
   cd your-repository-name
Create and activate a virtual environment:

sh
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages:

sh
Copy code
pip install -r requirements.txt
Run the scripts:

Extract and chunk the content from PDFs:

sh
Copy code
python extract_and_chunk.py
Embed the chunks and cluster them:

sh
Copy code
python embedding_and_cluster.py
Insert embeddings into Milvus and perform query retrieval:

sh
Copy code
python query_retrieval.py
Files and Directories
extract_and_chunk.py: Contains the code for extracting and chunking text from the selected textbooks.
embedding_and_cluster.py: Contains the code for embedding the text chunks and clustering them.
query_retrieval.py: Contains the code for inserting embeddings into Milvus and retrieving answers to questions.
requirements.txt: Lists the required Python libraries for the project.
data/: Directory where the extracted text chunks and embeddings are saved.
Dependencies
nltk
tqdm
PyPDF2
sentence-transformers
numpy
scikit-learn
transformers
torch
pymilvus
Make sure to install these dependencies using the provided requirements.txt file.

How to Use
Extract and Chunk Text:

Run the extract_and_chunk.py script to extract text from the selected PDFs and chunk the text into manageable pieces.
Embed and Cluster Chunks:

Run the embedding_and_cluster.py script to embed the text chunks using SBERT and cluster them using Gaussian Mixture Models.
Insert Embeddings and Retrieve Answers:

Run the query_retrieval.py script to insert the embeddings into Milvus and perform query retrieval.
Acknowledgments
Steps AI for providing the assessment task.
Milvus for the vector database.
Hugging Face for the transformers library.
