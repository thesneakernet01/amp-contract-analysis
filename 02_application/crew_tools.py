import os
from typing import Dict, List, Any, Optional
import chromadb
from langchain_openai import ChatOpenAI
# Import the tool decorator from crewai.tools
from crewai.tools import tool

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


# Define tools using the @tool decorator as per official documentation
@tool("ChromaDB Search Tool")
def search_chromadb(query: str, db_path: str = "/home/cdsw/02_application/chromadb",
                    collection_name: str = "document_chunks", n_results: int = 5) -> str:
    """Search for documents in ChromaDB based on queries.

    Args:
        query: The search query
        db_path: Path to ChromaDB
        collection_name: Name of the collection
        n_results: Number of results to return

    Returns:
        Formatted search results
    """
    try:
        # Make sure the DB path exists
        import os
        os.makedirs(db_path, exist_ok=True)

        # Initialize client with more explicit settings
        try:
            # Try the new API first
            client = chromadb.PersistentClient(path=db_path)
            print(f"Successfully connected to ChromaDB at {db_path}")
        except Exception as e1:
            print(f"Error with new ChromaDB API: {e1}")
            # Fall back to older API if needed
            try:
                client = chromadb.Client(chromadb.Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=db_path
                ))
                print(f"Successfully connected to ChromaDB using fallback method")
            except Exception as e2:
                print(f"Error with fallback ChromaDB connection: {e2}")
                return f"Could not connect to ChromaDB: {e1} / {e2}"

        # Try to get collection with error handling
        try:
            # Check if collection exists first
            all_collections = client.list_collections()
            collection_exists = any(c.name == collection_name for c in all_collections)

            if collection_exists:
                collection = client.get_collection(name=collection_name)
                print(f"Retrieved existing collection: {collection_name}")
            else:
                # Collection doesn't exist, create it
                print(f"Collection {collection_name} not found, creating it")
                # Try different embedding approaches
                try:
                    # First try SentenceTransformer
                    from chromadb.utils import embedding_functions
                    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2"
                    )
                    collection = client.create_collection(
                        name=collection_name,
                        embedding_function=ef
                    )
                except Exception as e:
                    print(f"Error with SentenceTransformer: {e}, using default embedding")
                    # Fall back to default embedding
                    collection = client.create_collection(name=collection_name)

                print(f"Created new collection: {collection_name}")
        except Exception as e:
            print(f"Error accessing or creating collection: {e}")
            return f"Error with ChromaDB collection: {e}"

        # Query the collection with better error handling
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )

            # Validate results structure
            if not results or "ids" not in results or not results["ids"]:
                return "No results found in ChromaDB for this query."

            # Format results for readability
            formatted_results = ""
            for i, doc_id in enumerate(results["ids"][0]):
                formatted_results += f"Document {i + 1}:\n"
                formatted_results += f"ID: {doc_id}\n"

                if "metadatas" in results and results["metadatas"] and len(results["metadatas"][0]) > i:
                    metadata = results["metadatas"][0][i]
                    for k, v in metadata.items():
                        formatted_results += f"{k}: {v}\n"

                if "documents" in results and results["documents"] and len(results["documents"][0]) > i:
                    doc_text = results["documents"][0][i]
                    formatted_results += f"Text: {doc_text[:500]}...\n\n"
                else:
                    formatted_results += "Text: [No text available]\n\n"

            return formatted_results
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return f"Error querying ChromaDB: {e}"
    except Exception as e:
        print(f"ChromaDB search error: {e}")
        return f"Error retrieving documents: {str(e)}"


@tool("Document Analysis Tool")
def analyze_text(query: str = None) -> str:
    """Analyze text using OpenAI LLM.

    Args:
        query: Optional query text

    Returns:
        Analysis results
    """
    global analysis_text, analysis_task

    # Get text from global variable if available
    text = analysis_text or query
    task = analysis_task or "analyze"

    if not text:
        return "No text provided for analysis"

    try:
        # Truncate text if needed
        max_length = 5000
        truncated_text = text[:max_length] if len(text) > max_length else text

        # Get API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No OpenAI API key found in environment")

        # Initialize LLM
        model = "gpt-4o"  # Default model
        try:
            # Try to get model from environment
            from main import get_document_processor
            processor = get_document_processor()
            if hasattr(processor, 'model'):
                model = processor.model
        except:
            pass

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

        # Reset the text to analyze to avoid memory issues
        reset_analysis_data()

        return response.content

    except Exception as e:
        print(f"Text analysis error: {e}")
        # Reset on error too
        reset_analysis_data()
        return f"Analysis failed: {str(e)}"


class ChromaDBRetrievalTool:
    """Backward compatibility wrapper for ChromaDBRetrievalTool."""

    def __init__(self, db_path: str = "/home/cdsw/02_application/chromadb",
                 collection_name: str = "document_chunks", **kwargs):
        """Initialize the ChromaDB retrieval tool wrapper."""
        self.name = "ChromaDBRetrievalTool"
        self.description = "Retrieves documents from ChromaDB based on queries"
        self.db_path = db_path
        self.collection_name = collection_name

        # Ensure the DB path exists during initialization
        import os
        os.makedirs(db_path, exist_ok=True)
        print(f"Initialized ChromaDBRetrievalTool with path: {os.path.abspath(db_path)}")

    # For backward compatibility
    def __call__(self, query):
        return search_chromadb(query, db_path=self.db_path, collection_name=self.collection_name)

class OllamaAnalysisTool:
    """Backward compatibility wrapper for OllamaAnalysisTool."""

    def __init__(self, max_length: int = 5000, model: str = "gpt-4o", **kwargs):
        """Initialize the analysis tool wrapper."""
        self.name = "OllamaAnalysisTool"
        self.description = "Analyzes text using OpenAI LLM"
        self.max_length = max_length
        self.model = model

    # Add setters for compatibility with main.py
    def set_text(self, text, task=None):
        """Set text and task for analysis."""
        set_text_for_analysis(text, task)

    def reset(self):
        """Reset text and task for analysis."""
        reset_analysis_data()

    # For backward compatibility
    def __call__(self, query=None):
        return analyze_text(query)