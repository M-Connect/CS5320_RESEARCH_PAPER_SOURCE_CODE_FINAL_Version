import fitz
from langchain_community.document_loaders import PyPDFLoader
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pinecone
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import chromadb
from pinecone import Pinecone, ServerlessSpec



# Constants
PDF_PATH = "RAG_Folder/InspectionReportPackage.pdf"
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530

# Load PDF and extract text
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text.
    """
    try:
        with fitz.open(pdf_path) as document:
            extracted_text = ""
            for page in document:
                extracted_text += page.get_text()
        return extracted_text
    except Exception as e:
        raise ValueError(f"Error processing PDF: {e}")

# Tokenize text
def tokenize_text(text: str) -> list[str]:
    """
    Tokenizes text into individual words.

    Args:
        text (str): Text to tokenize.

    Returns:
        list[str]: Tokenized text.
    """
    return nltk.word_tokenize(text)

# Remove stopwords
def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Removes stopwords from tokenized text.

    Args:
        tokens (list[str]): Tokenized text.

    Returns:
        list[str]: Text with stopwords removed.
    """
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t.lower() not in stop_words]

# Lemmatize text
def lemmatize_text(tokens: list[str]) -> list[str]:
    """
    Lemmatizes tokenized text.

    Args:
        tokens (list[str]): Tokenized text.

    Returns:
        list[str]: Lemmatized text.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]

# Remove special characters and punctuation
def remove_special_chars(tokens: list[str]) -> list[str]:
    """
    Removes special characters and punctuation from tokenized text.

    Args:
        tokens (list[str]): Tokenized text.

    Returns:
        list[str]: Text with special characters and punctuation removed.
    """
    return [re.sub(r'[^a-zA-Z0-9]', '', t) for t in tokens]

# Remove short words
def remove_short_words(tokens: list[str]) -> list[str]:
    """
    Removes short words from tokenized text.

    Args:
        tokens (list[str]): Tokenized text.

    Returns:
        list[str]: Text with short words removed.
    """
    return [t for t in tokens if len(t) >= 3]

# Convert to lowercase
def convert_to_lowercase(tokens: list[str]) -> list[str]:
    """
    Converts tokenized text to lowercase.

    Args:
        tokens (list[str]): Tokenized text.

    Returns:
        list[str]: Text in lowercase.
    """
    return [t.lower() for t in tokens]

# Preprocess text
def preprocess_text(text: str) -> str:
    """
    Preprocesses text by tokenizing, removing stopwords, lemmatizing, removing special characters, removing short words, and converting to lowercase.

    Args:
        text (str): Text to preprocess.

    Returns:
        str: Preprocessed text.
    """
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_text(tokens)
    tokens = remove_special_chars(tokens)
    tokens = remove_short_words(tokens)
    tokens = convert_to_lowercase(tokens)
    return ' '.join(tokens)

# Generate embeddings
def generate_embeddings(text: str) -> torch.Tensor:
    """
    Generates embeddings from preprocessed text using a transformer model.

    Args:
        text (str): Preprocessed text.

    Returns:
        torch.Tensor: Embeddings vector.
    """
    # Load pre-trained model and tokenizer
    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Use the CLS token

    # Return embeddings vector
    return embeddings

# Storing Embeddings in Vector Databases
def store_embeddings(embeddings, index_name):
    api_key = "pcsk_22grZC_Rqd7LfJLenCcdfdg28VqyPfVwaDwrhQZmNYFY9CZvgDjp6dfGLXstkdPGpM4NWZ"
    pinecone = Pinecone(api_key=api_key)

    if index_name not in pinecone.list_indexes().names():
        pinecone.create_index(
            name=index_name,
            dimension=768,
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    # Get the index
    index = pinecone.Index(index_name)

    # Convert PyTorch tensor to list of vectors
    embeddings_list = embeddings.numpy().tolist()[0]  # Extract single vector

    # Upsert vector with ID
    index.upsert(vectors=[embeddings_list], ids=["doc-1"])  # Replace "doc-1" with desired ID

# Main function
def main():
    text = extract_text_from_pdf(PDF_PATH)
    preprocessed_text = preprocess_text(text)
    embeddings = generate_embeddings(preprocessed_text)
    print(embeddings.shape)

    # Store embeddings in Pinecone
    index_name = "software-design"
    try:
        store_embeddings(embeddings, index_name)
    except pinecone.core.openapi.shared.exceptions.PineconeApiException as e:
        print(f"Pinecone API error: {e}")
    except AttributeError as e:
        print(f"Attribute error: {e}")

if __name__ == "__main__":
    main()