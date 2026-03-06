import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


class IngestionPipeline:

    def __init__(self, data_path):

        self.df = pd.read_csv(data_path)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Use only name for embeddings (less noise)
        self.documents = (
        self.df["name"].astype(str) 
        + self.df["description"].astype(str)
        ).tolist()
        self.embeddings = self.model.encode(
            self.documents,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        self.chroma_client = chromadb.PersistentClient(path="data/chroma_db")

        try:
            self.collection = self.chroma_client.get_collection("assessments")
            print("Loaded existing ChromaDB")

        except:

            print("Creating embeddings...")

            self.collection = self.chroma_client.create_collection(
                name="assessments"
            )

            self.collection.add(
                ids=[str(i) for i in range(len(self.documents))],
                embeddings=self.embeddings.tolist()
            )

    def get_collection(self):
        return self.collection

    def get_dataframe(self):
        return self.df

    def get_documents(self):
        return self.documents

    def get_embeddings(self):
        return self.embeddings

    def get_model(self):
        return self.model