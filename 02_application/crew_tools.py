import os
from typing import Dict, List, Any, Optional
import chromadb
from langchain_openai import ChatOpenAI


# Simplified tools for CrewAI compatibility
# Instead of using inheritance, we'll create simple functions that can be wrapped by CrewAI tools

def search_chromadb(query: str, db_path: str = "/home/cdsw/02_application/chromadb",
                    collection_name: str = "document_chunks", n_results: int = 5) -> str:
    """Search for documents in ChromaDB."""
    try:
        # Initialize client and collection
        client = chromadb.PersistentClient(path=db_path)
        try:
            collection = client.get_collection(name=collection_name)
            print(f"Retrieved existing collection: {collection_name}")
        except:
            # Use embedding function
            try:
                from chromadb.utils import embedding_functions
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            except:
                print("Using default embedding function")
                from chromadb.utils import embedding_functions
                ef = embedding_functions.DefaultEmbeddingFunction()

            print(f"Collection {collection_name} not found, creating it")
            collection = client.create_collection(
                name=collection_name,
                embedding_function=ef
            )

        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        # Format results for readability
        formatted_results = ""
        if results and "ids" in results and len(results["ids"]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                doc_text = results["documents"][0][i] if "documents" in results and len(
                    results["documents"]) > 0 else "No text available"
                metadata = results["metadatas"][0][i] if "metadatas" in results and len(
                    results["metadatas"]) > 0 else {}

                formatted_results += f"Document {i + 1}:\n"
                formatted_results += f"ID: {doc_id}\n"
                for k, v in metadata.items():
                    formatted_results += f"{k}: {v}\n"
                formatted_results += f"Text: {doc_text[:500]}...\n\n"
        else:
            formatted_results = "No results found in ChromaDB for this query."

        return formatted_results
    except Exception as e:
        print(f"ChromaDB search error: {e}")
        return f"Error retrieving documents: {str(e)}"


def analyze_text(text: str, task: str = "analyze", model: str = "gpt-4o", max_length: int = 5000) -> str:
    """Analyze text using OpenAI."""
    try:
        # Truncate text if needed
        truncated_text = text[:max_length] if len(text) > max_length else text

        # Get API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No OpenAI API key found in environment")

        # Initialize LLM
        llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=0.2
        )

        # Create the prompt based on task
        if task == "analyze":
            messages = [
                {"role": "system",
                 "content": "You are a legal document analyzer. Analyze the following document highlighting key legal provisions, rights, obligations, and potential legal issues."},
                {"role": "user", "content": truncated_text}
            ]
        elif task == "summarize":
            messages = [
                {"role": "system",
                 "content": "You are a legal document summarizer. Summarize the following document focusing on key legal points."},
                {"role": "user", "content": truncated_text}
            ]
        elif task == "extract_definitions":
            messages = [
                {"role": "system",
                 "content": "You are a legal terminology expert. Extract and explain all defined terms in the following document."},
                {"role": "user", "content": truncated_text}
            ]
        elif task == "assess_risks":
            messages = [
                {"role": "system",
                 "content": "You are a legal risk assessor. Identify and evaluate all legal risks in the following document, rating them by severity and likelihood."},
                {"role": "user", "content": truncated_text}
            ]
        elif task == "compare":
            messages = [
                {"role": "system",
                 "content": "You are a legal document comparison specialist. Compare the following documents, identifying key similarities and differences in legal provisions."},
                {"role": "user", "content": truncated_text}
            ]
        else:
            messages = [
                {"role": "system",
                 "content": f"You are a legal document specialist. Perform {task} on the following document."},
                {"role": "user", "content": truncated_text}
            ]

        # Get response
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        print(f"Text analysis error: {e}")
        return f"Analysis failed: {str(e)}"


# Global variables to store text and task for analysis tool
analysis_text = None
analysis_task = None


# Helper functions that will be used by the main application
def set_text_for_analysis(text, task=None):
    """Set text and task for analysis."""
    global analysis_text, analysis_task
    analysis_text = text
    analysis_task = task
    print(f"Set text for analysis, task: {task}")


def get_text_for_analysis():
    """Get text for analysis."""
    global analysis_text
    return analysis_text


def get_task_for_analysis():
    """Get task for analysis."""
    global analysis_task
    return analysis_task


def reset_analysis_data():
    """Reset text and task for analysis."""
    global analysis_text, analysis_task
    analysis_text = None
    analysis_task = None
    print("Reset analysis data")


# Functions that will be wrapped by CrewAI tools
def chromadb_tool(query: str) -> str:
    """Tool function for CrewAI to search ChromaDB."""
    return search_chromadb(query)


def analysis_tool(query: str = None) -> str:
    """Tool function for CrewAI to analyze text."""
    global analysis_text, analysis_task
    text = analysis_text or query
    task = analysis_task or "analyze"

    if not text:
        return "No text provided for analysis"

    result = analyze_text(text, task)

    # Reset after use
    analysis_text = None
    analysis_task = None

    return result