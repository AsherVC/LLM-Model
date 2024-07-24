from sklearn.mixture import GaussianMixture
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModel
import torch  # Import PyTorch

# Initialize the tokenizer and model for embeddings
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
model = AutoModel.from_pretrained(embed_model_id).to('cuda' if torch.cuda.is_available() else 'cpu')  # Move model to GPU if available

# Function to compute embeddings
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')  # Move tensors to GPU if available
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Move results back to CPU

# Load the embeddings and chunks
print("Loading embeddings and chunks...")
embeddings = np.load("embeddings.npy")
with open("chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.read().splitlines()

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)  # Use GPU for summarization if available

# Function to summarize text using the BART model
def summarize_text(text, max_chunk_length=1024):
    text_chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    summaries = []
    for chunk in text_chunks:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return ' '.join(summaries)

# Function to cluster and summarize embeddings
def cluster_and_summarize(embeddings, n_clusters=10):
    print("Clustering embeddings...")
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='tied')
    gmm.fit(embeddings)
    cluster_labels = gmm.predict(embeddings)

    cluster_texts = [""] * n_clusters
    for i, label in enumerate(cluster_labels):
        cluster_texts[label] += chunks[i] + " "

    print("Summarizing clusters...")
    from tqdm import tqdm
    summaries = [summarize_text(cluster_text) for cluster_text in tqdm(cluster_texts, desc="Summarizing cluster texts")]
    return summaries, gmm

# Cluster and summarize embeddings
n_clusters = 10  # Adjust the number of clusters as needed
summaries, gmm = cluster_and_summarize(embeddings, n_clusters)

# Save summaries and GMM model
np.save("summaries.npy", summaries)
import joblib
joblib.dump(gmm, "gmm_model.pkl")
