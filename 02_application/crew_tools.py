import os
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from crewai.tools import tool
from langchain_openai import ChatOpenAI

# Global variables to store text and task for analysis tool
analysis_text = None
analysis_task = None


class ChromaDBStorage:
    """Singleton class to manage ChromaDB connections."""
    _instance = None

    def __new__(cls, db_path: str = "./chromadb", chunks_collection: str = "document_chunks",
                summaries_collection: str = "document_summaries"):
        if cls._instance is None:
            cls._instance = super(ChromaDBStorage, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: str = "./chromadb", chunks_collection: str = "document_chunks",
                 summaries_collection: str = "document_summaries"):
        """Initialize ChromaDB storage as a singleton to avoid multiple connections."""
        if self._initialized:
            return

        print(f"Initializing ChromaDB storage at {os.path.abspath(db_path)}")
        print(f"Chunks collection: {chunks_collection}")
        print(f"Summaries collection: {summaries_collection}")

        self.db_path = db_path
        self.chunks_collection_name = chunks_collection
        self.summaries_collection_name = summaries_collection

        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        print(f"ChromaDB directory exists: {os.path.exists(db_path)}")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        print("ChromaDB client initialized")

        # Create embedding function
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        print("Embedding function created")

        # Get or create chunks collection
        try:
            self.chunks_collection = self.client.get_collection(
                name=chunks_collection,
                embedding_function=self.ef
            )
            print(f"Retrieved existing chunks collection: {chunks_collection}")
        except:
            print(f"Chunks collection not found, creating: {chunks_collection}")
            self.chunks_collection = self.client.create_collection(
                name=chunks_collection,
                embedding_function=self.ef
            )
            print(f"Created new chunks collection: {chunks_collection}")

        # Get or create summaries collection
        try:
            self.summaries_collection = self.client.get_collection(
                name=summaries_collection,
                embedding_function=self.ef
            )
            print(f"Retrieved existing summaries collection: {summaries_collection}")
        except:
            print(f"Summaries collection not found, creating: {summaries_collection}")
            self.summaries_collection = self.client.create_collection(
                name=summaries_collection,
                embedding_function=self.ef
            )
            print(f"Created new summaries collection: {summaries_collection}")

        self._initialized = True


# Helper functions for text analysis
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


# Get the ChromaDB storage instance
def get_storage(db_path: str = "./chromadb"):
    """Get the ChromaDB storage instance."""
    return ChromaDBStorage(db_path=db_path)


# CrewAI tool functions using the decorator pattern
@tool("Add Texts to ChromaDB")
def add_texts(texts: List[str], metadatas: List[Dict], ids: List[str], db_path: str = "./chromadb",
              collection_name: str = "document_chunks") -> str:
    """Add texts to ChromaDB.

    Args:
        texts: List of text chunks to add
        metadatas: List of metadata dictionaries for each chunk
        ids: List of IDs for each chunk
        db_path: Path to ChromaDB
        collection_name: Name of the collection

    Returns:
        Status message
    """
    try:
        storage = get_storage(db_path)

        if collection_name == "document_chunks":
            collection = storage.chunks_collection
        else:
            # Handle using a different collection
            try:
                collection = storage.client.get_collection(name=collection_name, embedding_function=storage.ef)
            except:
                collection = storage.client.create_collection(name=collection_name, embedding_function=storage.ef)

        print(f"Adding {len(texts)} chunks to ChromaDB collection {collection_name}")

        # Add texts to collection
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Successfully added {len(texts)} chunks to ChromaDB")

        # Verify addition
        result = collection.get()
        print(f"Collection now contains {len(result['ids'])} items")

        return f"Added {len(texts)} chunks to ChromaDB collection {collection_name}"
    except Exception as e:
        print(f"Error adding texts to ChromaDB: {e}")
        return f"Error: {str(e)}"


@tool("Add Summary to ChromaDB")
def add_summary(doc_id: str, source: str, filename: str, summary: str, analysis: str,
                db_path: str = "./chromadb", summaries_collection: str = "document_summaries") -> str:
    """Add document summary to ChromaDB.

    Args:
        doc_id: Document ID
        source: Source file path
        filename: Original filename
        summary: Document summary text
        analysis: Document analysis text
        db_path: Path to ChromaDB
        summaries_collection: Name of the summaries collection

    Returns:
        Status message
    """
    try:
        storage = get_storage(db_path)

        # Get the summaries collection
        try:
            collection = storage.client.get_collection(name=summaries_collection, embedding_function=storage.ef)
        except:
            collection = storage.client.create_collection(name=summaries_collection, embedding_function=storage.ef)

        print(f"Adding summary for document {doc_id} to ChromaDB")

        # Create a combined document with summary and analysis
        combined_text = f"SUMMARY:\n{summary}\n\nANALYSIS:\n{analysis}"

        # Create metadata
        metadata = {
            "doc_id": doc_id,
            "source": source,
            "filename": filename,
            "type": "summary_analysis"
        }

        # Create ID for the summary
        summary_id = f"{doc_id}_summary"

        # Add summary to collection
        collection.add(
            documents=[combined_text],
            metadatas=[metadata],
            ids=[summary_id]
        )

        print(f"Successfully added summary for document {doc_id} to ChromaDB")

        return f"Added summary for document {doc_id} to ChromaDB collection {summaries_collection}"
    except Exception as e:
        print(f"Error adding summary to ChromaDB: {e}")
        return f"Error: {str(e)}"


@tool("Search ChromaDB Chunks")
def search_chunks(query: str, n_results: int = 5, db_path: str = "./chromadb",
                  collection_name: str = "document_chunks") -> str:
    """Search for similar texts in ChromaDB chunks collection.

    Args:
        query: Search query
        n_results: Number of results to return
        db_path: Path to ChromaDB
        collection_name: Name of the collection

    Returns:
        Formatted search results
    """
    try:
        storage = get_storage(db_path)

        print(f"Searching for chunks with query: {query}")

        # Get the collection
        try:
            collection = storage.client.get_collection(name=collection_name, embedding_function=storage.ef)
        except:
            return f"Collection {collection_name} not found"

        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        print(f"Found {len(results['ids'][0]) if results['ids'] and len(results['ids']) > 0 else 0} matching chunks")

        # Format results
        formatted_results = ""
        if results and "ids" in results and len(results["ids"]) > 0 and len(results["ids"][0]) > 0:
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
        else:
            return "No results found in ChromaDB for this query."
    except Exception as e:
        print(f"Error searching chunks: {e}")
        return f"Error: {str(e)}"


@tool("Search ChromaDB Summaries")
def search_summaries(query: str, n_results: int = 5, db_path: str = "./chromadb",
                     collection_name: str = "document_summaries") -> str:
    """Search for similar summaries in ChromaDB.

    Args:
        query: Search query
        n_results: Number of results to return
        db_path: Path to ChromaDB
        collection_name: Name of the collection

    Returns:
        Formatted search results
    """
    try:
        storage = get_storage(db_path)

        print(f"Searching for summaries with query: {query}")

        # Get the collection
        try:
            collection = storage.client.get_collection(name=collection_name, embedding_function=storage.ef)
        except:
            return f"Collection {collection_name} not found"

        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        print(f"Found {len(results['ids'][0]) if results['ids'] and len(results['ids']) > 0 else 0} matching summaries")

        # Format results
        formatted_results = ""
        if results and "ids" in results and len(results["ids"]) > 0 and len(results["ids"][0]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted_results += f"Document {i + 1}:\n"
                formatted_results += f"ID: {doc_id}\n"

                if "metadatas" in results and results["metadatas"] and len(results["metadatas"][0]) > i:
                    metadata = results["metadatas"][0][i]
                    for k, v in metadata.items():
                        formatted_results += f"{k}: {v}\n"

                if "documents" in results and results["documents"] and len(results["documents"][0]) > i:
                    doc_text = results["documents"][0][i]
                    formatted_results += f"Summary: {doc_text[:200]}...\n\n"
                else:
                    formatted_results += "Summary: [No text available]\n\n"

            return formatted_results
        else:
            return "No results found in ChromaDB for this query."
    except Exception as e:
        print(f"Error searching summaries: {e}")
        return f"Error: {str(e)}"


@tool("Get All Summaries")
def get_all_summaries(db_path: str = "./chromadb", collection_name: str = "document_summaries") -> str:
    """Get all document summaries from ChromaDB.

    Args:
        db_path: Path to ChromaDB
        collection_name: Name of the collection

    Returns:
        Formatted list of all summaries
    """
    try:
        storage = get_storage(db_path)

        print(f"Retrieving all document summaries")

        # Get the collection
        try:
            collection = storage.client.get_collection(name=collection_name, embedding_function=storage.ef)
        except:
            return f"Collection {collection_name} not found"

        # Get all documents
        results = collection.get()

        print(f"Retrieved {len(results['ids'])} summaries")

        # Format results
        formatted_results = f"Found {len(results['ids'])} document summaries:\n\n"

        for i, doc_id in enumerate(results["ids"]):
            formatted_results += f"Document {i + 1}:\n"
            formatted_results += f"ID: {doc_id}\n"

            if "metadatas" in results and i < len(results["metadatas"]):
                metadata = results["metadatas"][i]
                formatted_results += f"Filename: {metadata.get('filename', 'Unknown')}\n"
                formatted_results += f"Source: {metadata.get('source', 'Unknown')}\n"

            formatted_results += "\n"

        return formatted_results
    except Exception as e:
        print(f"Error getting all summaries: {e}")
        return f"Error: {str(e)}"


@tool("Get Document Summary")
def get_summary(doc_id: str, db_path: str = "./chromadb", collection_name: str = "document_summaries") -> str:
    """Get summary for a specific document from ChromaDB.

    Args:
        doc_id: Document ID
        db_path: Path to ChromaDB
        collection_name: Name of the collection

    Returns:
        Document summary
    """
    try:
        storage = get_storage(db_path)

        print(f"Retrieving summary for document {doc_id}")

        # Get the collection
        try:
            collection = storage.client.get_collection(name=collection_name, embedding_function=storage.ef)
        except:
            return f"Collection {collection_name} not found"

        # Get the summary
        summary_id = f"{doc_id}_summary"
        result = collection.get(ids=[summary_id])

        if result["ids"]:
            print(f"Found summary for document {doc_id}")

            # Format result
            formatted_result = f"Summary for document {doc_id}:\n\n"

            if "metadatas" in result and result["metadatas"]:
                metadata = result["metadatas"][0]
                formatted_result += f"Filename: {metadata.get('filename', 'Unknown')}\n"
                formatted_result += f"Source: {metadata.get('source', 'Unknown')}\n\n"

            if "documents" in result and result["documents"]:
                formatted_result += result["documents"][0]
            else:
                formatted_result += "[No summary available]"

            return formatted_result
        else:
            return f"No summary found for document {doc_id}"
    except Exception as e:
        print(f"Error getting summary: {e}")
        return f"Error: {str(e)}"


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
            # Try to get model from environment or configuration
            from main import OllamaDocumentProcessor
            processor = OllamaDocumentProcessor()
            if hasattr(processor, 'model_name'):
                model = processor.model_name
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


# For backward compatibility with main.py
class ChromaDBRetrievalTool:
    """Backward compatibility wrapper for ChromaDBRetrievalTool."""

    def __init__(self, db_path: str = "./chromadb", collection_name: str = "document_chunks", **kwargs):
        """Initialize the ChromaDB retrieval tool wrapper."""
        self.name = "ChromaDBRetrievalTool"
        self.description = "Retrieves documents from ChromaDB based on queries"
        self.db_path = db_path
        self.collection_name = collection_name

        # Initialize the storage
        get_storage(db_path)

    def __call__(self, query):
        """Handle direct calls for backward compatibility."""
        return search_chunks(query=query, db_path=self.db_path, collection_name=self.collection_name)

    def get_function(self):
        """Return the function for newer versions of CrewAI."""
        return search_chunks


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

    # For backward compatibility, when this object is called directly
    def __call__(self, query=None):
        return analyze_text(query)

    # For use with newer versions of crewai
    def get_function(self):
        return analyze_text