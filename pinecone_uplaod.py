import os
import pandas as pd
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

def load_env():
    load_dotenv()  # loads variables from .env in current folder

def init_pinecone(api_key, index_name):
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    return index

def upsert_data_with_sentence_transformers(index, texts, model, language, batch_size=100):
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
        for j, vector in enumerate(batch_embeddings):
            vectors.append({
                "id": f"{language}_{i+j}",
                "values": vector.tolist(),  # Pinecone expects lists, not numpy arrays
                "metadata": {
                    "text": batch_texts[j],
                    "language": language
                }
            })
        index.upsert(vectors=vectors)
        print(f"Upserted batch of {len(vectors)} for {language}")
        vectors = []
    if vectors:
        index.upsert(vectors=vectors)
        print(f"Upserted last batch of {len(vectors)} for {language}")

def query_pinecone(index, query_text, model, language, top_k=5):
    query_vector = model.encode([query_text], convert_to_numpy=True)[0].tolist()
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter={"language": language}
    )
    return response

def main():
    load_env()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = "customer-support-chat"  # your Pinecone index name

    # Initialize Pinecone index
    index = init_pinecone(PINECONE_API_KEY, INDEX_NAME)

    # Load datasets
    df_en = pd.read_csv("english_dataset.csv")    # has 'combined' column
    df_hi = pd.read_csv("hindi_dataset.csv")      # has 'combined_hi' column

    # Initialize sentence-transformers models
    en_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    hi_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("Uploading English data...")
    upsert_data_with_sentence_transformers(index, df_en['combined'].tolist(), en_model, "english")

    print("Uploading Hindi data...")
    upsert_data_with_sentence_transformers(index, df_hi['combined_hi'].tolist(), hi_model, "hindi")

    # Test query (English)
    test_query_en = "How can I reset my password?"
    print(f"\nQuerying English for: {test_query_en}")
    res_en = query_pinecone(index, test_query_en, en_model, "english")
    for match in res_en['matches']:
        print(f"Score: {match['score']:.3f} | Text: {match['metadata']['text']}")

    # Test query (Hindi)
    test_query_hi = "मैं अपना पासवर्ड कैसे रीसेट कर सकता हूँ?"
    print(f"\nQuerying Hindi for: {test_query_hi}")
    res_hi = query_pinecone(index, test_query_hi, hi_model, "hindi")
    for match in res_hi['matches']:
        print(f"Score: {match['score']:.3f} | Text: {match['metadata']['text']}")

if __name__ == "__main__":
    main()
