# Textbook Content Extraction and Question Answering System

This project is a take-home assessment for an NLP Intern position at Steps AI. The objective is to create a question-answering system based on the content of three textbooks.

## Project Overview

The project involves the following key steps:

1. **Extracting Content from Textbooks:**
   - Select three digital textbooks with more than 300 pages each.
   - Extract content from the selected textbooks thoroughly.

2. **Data Chunking and Embedding:**
   - Chunk the extracted content into short, contiguous texts of approximately 100 tokens each.
   - Embed the chunked texts using Sentence-BERT (SBERT).

3. **RAPTOR Indexing:**
   - Cluster the embedded chunks using Gaussian Mixture Models (GMMs) with soft clustering.
   - Summarize the clusters using a Language Model (e.g., GPT-3.5-turbo).
   - Re-embed the summarized texts and recursively apply the clustering and summarization process until a hierarchical tree structure is formed.
   - Store the RAPTOR index in a Milvus vector database.

4. **Retrieval Techniques:**
   - Implement query expansion techniques to enhance the retrieval process.
   - Employ hybrid retrieval methods combining BM25 and BERT/bi-encoder based retrieval methods.
   - Re-rank the retrieved data based on relevance and similarity to the query.

5. **Question Answering:**
   - Pass the retrieved and re-ranked data to a Language Model for generating accurate and relevant answers based on the retrieved content.

## Setup and Installation

### 1. Clone the Repository

```sh
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```
### 2. Create and Activate a Virtual Environment

```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Required Packages

```sh
pip install -r requirements.txt
```

### 4. Run the Scripts

#### Extract and Chunk the Content from PDFs:
```sh
python main.py
```

#### Embed the Chunks and Cluster Them:
```sh
python summerize.py
```
#### Insert Embeddings into Milvus and Perform Query Retrieval:
```sh
python Query_Retrieval.py
```

## Dependencies
- nltk
- tqdm
- PyPDF2
- sentence-transformers
- numpy
- scikit-learn
- transformers
- torch
- pymilvus
Make sure to install these dependencies using the provided requirements.txt file.

## How to Use
### Extract and Chunk Text:
Run the `main.py` script to extract text from the selected PDFs and chunk the text into manageable pieces.

## Embed and Cluster Chunks:
Run the `summerize.py` script to embed the text chunks using SBERT and cluster them using Gaussian Mixture Models.

## Insert Embeddings and Retrieve Answers:
Run the `Query_Retrieval.py` script to insert the embeddings into Milvus and perform query retrieval.

## Acknowledgments
- Steps AI for providing the assessment task.
- Milvus for the vector database.
- Hugging Face for the transformers library.


