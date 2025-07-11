from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import extract
def create_faiss_index(documents, save_path="faiss_index"):
    print(f"Loaded {len(documents)} documents")
    
    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks")
    
    # Use HuggingFaceEmbeddings from langchain-huggingface package
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create FAISS index
    index = FAISS.from_documents(docs, embeddings)
    
    # Save index locally
    index.save_local(save_path)
    print(f"FAISS index saved to {save_path}")
    
    return index
a=extract.run()
# Example usage:
index = create_faiss_index(a)
