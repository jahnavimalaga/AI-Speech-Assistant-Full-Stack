import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import SentenceTransformer
from groq import Groq
from api_keys import user_data


class PDFProcessor:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def process_by_page(self):
        data = []
        for root, _, files in os.walk(self.directory_path):
            for file in files:
                if not file.lower().endswith(".pdf"):
                    print(f"Skipping non-PDF file: {file}")
                    continue
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                loader = PyPDFLoader(file_path)
                pages = loader.load_and_split()
                for i, page in enumerate(pages):
                    data.append({
                        "File": file_path,
                        "Page": i + 1,
                        "Data": page.page_content
                    })
        return data


class DocumentPreparer:
    @staticmethod
    def prepare(documents):
        prepared_docs = []
        for entry in documents:
            file_path = entry['File']
            page_number = entry['Page']
            content = entry['Data']
            file_name = os.path.basename(file_path)
            folder_names = file_path.split("/")[2:-1]

            doc = Document(
                page_content=f"<Source>\n{file_path}\n</Source>\n\n<Content>\n{content}\n</Content>",
                metadata={
                    "file_name": file_name,
                    "parent_folder": folder_names[-1] if folder_names else "",
                    "folder_names": folder_names,
                    "page_number": page_number
                }
            )
            prepared_docs.append(doc)
        return prepared_docs


class EmbeddingEngine:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text):
        return self.model.encode(text)

    def cosine_similarity(self, sentence1, sentence2):
        emb1 = self.get_embedding(sentence1).reshape(1, -1)
        emb2 = self.get_embedding(sentence2).reshape(1, -1)
        return cosine_similarity(emb1, emb2)[0][0]


class RAGPipeline:
    def __init__(self, pinecone_api_key, index_name, namespace, model_name="sentence-transformers/all-mpnet-base-v2", use_groq=True):
        self.embeddings = EmbeddingEngine(model_name)
        self.pinecone = Pinecone(api_key=pinecone_api_key)
        self.index = self.pinecone.Index(index_name)
        self.namespace = namespace
        self.use_groq = use_groq

        if use_groq:
            groq_api_key = user_data.get("GROQ_API_KEY")
            self.groq_client = Groq(api_key=groq_api_key)
        else:
            # Optional: Initialize OpenAI client here
            pass

    def query(self, query_text):
        query_vector = self.embeddings.get_embedding(query_text)
        top_matches = self.index.query(
            vector=query_vector.tolist(),
            top_k=10,
            include_metadata=True,
            namespace=self.namespace
        )

        contexts = [m['metadata']['text'] for m in top_matches['matches']]
        augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query_text

        system_prompt = (
            "You are an expert at understanding and analyzing ayurvedic data - particularly in giving advice "
            "on what medicine to take for what health issue.\n\n"
            "Answer any questions I have, based on the document provided. Always consider all parts of the context."
        )

        if self.use_groq:
            res = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": augmented_query}
                ]
            )
            return res.choices[0].message.content

        else:
            # OpenAI version here if needed
            pass


class RAGSystem:
    def __init__(self, directory_path, index_name, namespace):
        pinecone_api_key = user_data.get("pinecone_api_key")
        os.environ['PINECONE_API_KEY'] = pinecone_api_key

        self.directory_path = directory_path
        self.processor = PDFProcessor(directory_path)
        self.index_name = index_name
        self.namespace = namespace

        # Embeddings and RAG pipeline
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.rag_pipeline = RAGPipeline(
            pinecone_api_key=pinecone_api_key,
            index_name=index_name,
            namespace=namespace
        )

    def index_documents(self):
        print("Processing and indexing documents into Pinecone...")
        documents = self.processor.process_by_page()
        document_data = DocumentPreparer.prepare(documents)

        # Insert documents into Pinecone
        PineconeVectorStore.from_documents(
            documents=document_data,
            embedding=self.embeddings,
            index_name=self.index_name,
            namespace=self.namespace
        )
        print(f"Indexed {len(document_data)} documents into Pinecone.")

    def run(self, query):
        response = self.rag_pipeline.query(query)
        return response


# ---- USAGE ---- #
if __name__ == "__main__":
    rag = RAGSystem(
        directory_path="data/book/",
        index_name="codebase-rag",
        namespace="aurvedic_medicine_pdf"
    )
      # Index your documents first
    # rag.index_documents()
    response = rag.run("What medicine should I take if I get fever?")
    print("Response:", response)
