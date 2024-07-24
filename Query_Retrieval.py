import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer

# Connect to Milvus
try:
    connections.connect()
except Exception as e:
    print(f"Failed to connect to Milvus server: {e}")
    exit(1)

# Define the collection name
collection_name = "text_embedding_collection"

# Define schema (make sure it matches the one used for creation)
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)  # Adjust `dim` if necessary
]
schema = CollectionSchema(fields, description="A collection of text embeddings")

# Create or get the collection
if not utility.has_collection(collection_name):
    collection = Collection(name=collection_name, schema=schema)
else:
    collection = Collection(name=collection_name)

# Load embeddings from .npy file
def load_embeddings(file_path):
    return np.load(file_path)

# Insert embeddings into Milvus
def insert_embeddings(collection, embeddings):
    # Ensure that 'embeddings' is a list of lists where each list contains one embedding vector
    if not isinstance(embeddings[0], list):
        embeddings = [list(embed) for embed in embeddings]  # Convert to list of lists if needed
    
    # Prepare the data to match the schema: [id_list, embeddings_list]
    ids = [None] * len(embeddings)  # Use None if ids are auto-generated
    data_to_insert = [ids, embeddings]  # Only include fields defined in the schema
    
    # Insert embeddings into the collection
    collection.insert(data_to_insert)
    print(f"Number of entities in Milvus: {collection.num_entities}")

# Ensure collection is loaded
def load_collection(collection):
    if not collection.is_loaded:
        collection.load()
        collection.wait_for_loading_complete()

# Search function
def retrieve_and_rank(query_embedding, collection):
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    
    # Ensure collection is loaded
    load_collection(collection)
    
    # Perform the search
    res = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=10
    )
    
    # Process search results
    return res

# Example function to get embedding for a question
def get_embedding_for_question(question):
    # Initialize the Sentence-BERT model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    question_embedding = model.encode([question])
    return question_embedding[0].tolist()

# Query function
def answer_question(question):
    # Convert the question to an embedding
    query_embedding = get_embedding_for_question(question)
    
    # Perform retrieval and ranking
    results = retrieve_and_rank(query_embedding, collection)
    
    # Process results
    for hit in results[0]:
        print(f"ID: {hit.id}, Distance: {hit.distance}")
    return results

# Load embeddings from file and insert into Milvus
embeddings = load_embeddings('embeddings.npy')
insert_embeddings(collection, embeddings)

# Example usage
question = "Your example question here"
answer_question(question)
