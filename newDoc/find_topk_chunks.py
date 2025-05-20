import spacy
import numpy as np
from typing import Dict, List
import os
import glob
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_spacy_model(lang_model: str = "en_core_web_sm") -> spacy.language.Language:
    """
    Load spaCy language model.
    Args:
        lang_model: Name of the spaCy model.
    Returns:
        Loaded spaCy model.
    """
    try:
        return spacy.load(lang_model)
    except OSError:
        raise ImportError(f"spaCy model '{lang_model}' not found. Install it using 'python -m spacy download {lang_model}'")

def load_documents_from_folder(folder_path: str) -> List[Dict]:
    """
    Load all .txt files from a folder.
    Args:
        folder_path: Path to the folder containing .txt files.
    Returns:
        List of dictionaries: [{file_path, content}, ...]
    """
    txt_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {folder_path}")
    documents = []
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            documents.append({
                "file_path": file_path,
                "content": content
            })
    return documents

def preprocess_documents(documents: List[Dict], nlp: spacy.language.Language, max_tokens: int = 100) -> List[Dict]:
    """
    Split documents into chunks based on a fixed token count, ensuring sentences are not split.
    Args:
        documents: List of dictionaries [{file_path, content}, ...]
        nlp: Loaded spaCy model
        max_tokens: Maximum number of tokens per chunk
    Returns:
        List of chunks: [{file_path, chunk_id, text}, ...]
    """
    chunks = []
    for doc in documents:
        file_path = doc["file_path"]
        doc_text = doc["content"]
        doc_nlp = nlp(doc_text)
        
        current_chunk = []
        current_token_count = 0
        chunk_id = 0
        
        for sent in doc_nlp.sents:
            sent_text = sent.text.strip()
            if not sent_text:  # Skip empty sentences
                continue
            sent_tokens = len(sent)
            
            # If adding this sentence exceeds max_tokens, finalize the current chunk
            if current_token_count + sent_tokens > max_tokens and current_chunk:
                chunks.append({
                    "file_path": file_path,
                    "chunk_id": chunk_id,
                    "text": " ".join(current_chunk).strip()
                })
                current_chunk = []
                current_token_count = 0
                chunk_id += 1
            
            # Add the sentence to the current chunk
            current_chunk.append(sent_text)
            current_token_count += sent_tokens
        
        # Add the final chunk if it exists
        if current_chunk:
            chunks.append({
                "file_path": file_path,
                "chunk_id": chunk_id,
                "text": " ".join(current_chunk).strip()
            })
    
    return chunks

def find_top_k_chunks(
    folder_path: str,
    attribute_dict: Dict[str, str],
    top_k: int = 5,
    bert_model: str = "all-MiniLM-L6-v2",
    lang_model: str = "en_core_web_sm",
    max_tokens_per_chunk: int = 100
) -> Dict[str, Dict[str, List[str]]]:
    """
    Find the top-K most relevant chunks for each attribute in each file based on BERT embeddings.
    Args:
        folder_path: Path to the folder containing .txt files.
        attribute_dict: Dictionary mapping attributes to their reference texts {attribute: reference}.
        top_k: Number of top chunks to return per attribute per file.
        bert_model: Name of the BERT model for embeddings.
        lang_model: spaCy language model for chunking.
        max_tokens_per_chunk: Maximum number of tokens per chunk.
    Returns:
        Dictionary: {file_path: {attribute: [top_k_chunks]}}
    """
    # Load spaCy model
    nlp = load_spacy_model(lang_model)

    # Load BERT model
    try:
        model = SentenceTransformer(bert_model)
    except Exception as e:
        raise ImportError(f"Failed to load BERT model '{bert_model}': {e}. Ensure sentence-transformers is installed.")

    # Load and preprocess documents
    documents = load_documents_from_folder(folder_path)
    chunks = preprocess_documents(documents, nlp, max_tokens_per_chunk)
    
    # Group chunks by file
    file_chunks = {}
    for chunk in chunks:
        file_path = chunk["file_path"]
        if file_path not in file_chunks:
            file_chunks[file_path] = []
        file_chunks[file_path].append(chunk["text"])
    
    # Prepare attribute references
    attributes = list(attribute_dict.keys())
    references = [attribute_dict[attr] for attr in attributes]
    
    # Generate embeddings for references
    try:
        reference_embeddings = model.encode(references, convert_to_numpy=True, show_progress_bar=False)
    except Exception as e:
        raise RuntimeError(f"Failed to generate BERT embeddings for references: {e}")
    
    # Initialize result dictionary
    result = {file_path: {attr: [] for attr in attributes} for file_path in file_chunks}
    
    # Process each file
    for file_path, chunks in file_chunks.items():
        if not chunks:
            continue
        
        # Generate embeddings for chunks
        try:
            chunk_embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            print(f"Warning: Failed to generate BERT embeddings for chunks in {file_path}: {e}")
            continue
        
        # Compute cosine similarities
        similarities = cosine_similarity(chunk_embeddings, reference_embeddings)
        
        # Find top-K chunks for each attribute
        for attr_idx, attr in enumerate(attributes):
            # Get similarities for this attribute
            attr_similarities = similarities[:, attr_idx]
            # Get indices of top-K similarities
            top_k_indices = np.argsort(attr_similarities)[-top_k:][::-1]
            # Ensure we don't exceed available chunks
            top_k_indices = top_k_indices[:min(top_k, len(chunks))]
            # Get the top-K chunks
            top_k_chunks = [chunks[idx] for idx in top_k_indices]
            result[file_path][attr] = top_k_chunks
    
    return result

def get_topk_chunks(
    folder_path: str,
    attribute_dict: Dict[str, str],
    top_k: int = 5,
    bert_model: str = "all-MiniLM-L6-v2",
    lang_model: str = "en_core_web_sm",
    max_tokens_per_chunk: int = 100
):
    """
    Main function to find top-K relevant chunks for each attribute in each file.
    Args:
        folder_path: Path to the folder containing .txt files.
        attribute_dict: Dictionary mapping attributes to their reference texts.
        top_k: Number of top chunks to return per attribute.
        bert_model: BERT model name.
        lang_model: spaCy language model.
        max_tokens_per_chunk: Maximum number of tokens per chunk.
    Returns:
        Dictionary: {file_path: {attribute: [top_k_chunks]}}
    """
    result = find_top_k_chunks(
        folder_path,
        attribute_dict,
        top_k,
        bert_model,
        lang_model,
        max_tokens_per_chunk
    )
    
    # Print results for verification
    print("\nTop-K Chunks for Each Attribute in Each File:")
    for file_path, attr_chunks in result.items():
        print(f"\nFile: {file_path}")
        for attr, chunks in attr_chunks.items():
            print(f"  Attribute: {attr}")
            if not chunks:
                print("    No chunks found")
            for i, chunk in enumerate(chunks, 1):
                print(f"    Top-{i} Chunk: {chunk}")
    
    return result