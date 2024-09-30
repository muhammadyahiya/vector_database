import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="path_to_store"))

# Create or get a collection
collection = client.create_collection(name="excel_data_collection")

# Initialize sentence-transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to read all sheets from an Excel file and convert to embeddings
def excel_to_chroma(file_path, collection):
    # Read all sheets
    excel_data = pd.read_excel(file_path, sheet_name=None)  # sheet_name=None reads all sheets
    
    # Iterate over each sheet
    for sheet_name, df in excel_data.items():
        print(f"Processing sheet: {sheet_name}")
        
        # Optional: Preprocess data if necessary (depends on the content and your use case)
        # For this example, assuming text data in a column called 'text'
        if 'text' in df.columns:
            texts = df['text'].astype(str).tolist()  # Convert column to list of strings
            
            # Generate embeddings for the text
            embeddings = model.encode(texts, convert_to_tensor=False)
            
            # Add to Chroma collection
            for i, embedding in enumerate(embeddings):
                collection.add(
                    ids=[f"{sheet_name}_{i}"],  # Use a unique ID (sheet_name + row number)
                    embeddings=[embedding],
                    metadatas=[{"sheet": sheet_name, "row": i, "text": texts[i]}]  # Store metadata
                )

# Convert Excel file to ChromaDB collection
excel_file = "path_to_your_excel_file.xlsx"
excel_to_chroma(excel_file, collection)

# Optionally: Persist the data
client.persist()
