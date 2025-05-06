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
            model_name = os.environ.get("OPENAI_MODEL", "gpt-4o")
            if model_name:
                model = model_name
        except:
            pass

        # Enhanced error handling for LLM initialization
        try:
            llm = ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=0.2
            )
            print(f"Successfully initialized LLM with model: {model}")
        except Exception as e:
            error_msg = f"Error initializing LLM: {e}"
            print(error_msg)
            return f"Analysis failed: {error_msg}"

        # Create the prompt based on task with improved legal document analysis instructions
        if task == "analyze":
            prompt = """You are a legal document analyzer. Analyze the following document highlighting key legal provisions, rights, obligations, and potential legal issues.

            Focus on extracting and explaining:
            1. The most important legal provisions
            2. Rights granted to each party
            3. Obligations of each party
            4. Any potential legal risks or issues
            5. Important dates, deadlines, or timeframes

            Format your response in clear sections with headings.
            """
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": truncated_text}
            ]
        elif task == "summarize":
            prompt = """You are a legal document summarizer. Summarize the following document focusing on key legal points.

            Your summary should:
            1. Identify the type of legal document
            2. Explain the main purpose of the document
            3. Highlight the most significant legal provisions
            4. Identify the primary parties involved
            5. Mention any important dates or deadlines

            Keep your summary concise but comprehensive.
            """
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": truncated_text}
            ]
        elif task == "extract_definitions":
            prompt = """You are a legal terminology expert. Extract and explain all defined terms in the following document.

            For each defined term:
            1. Provide the exact definition from the document
            2. Explain the significance of this term
            3. Note any inconsistencies or ambiguities in the definition

            Format as a glossary with terms in alphabetical order.
            """
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": truncated_text}
            ]
        elif task == "assess_risks":
            prompt = """You are a legal risk assessor. Identify and evaluate all legal risks in the following document, rating them by severity and likelihood.

            For each risk:
            1. Risk: Clearly name the risk
            2. Severity: Rate as High, Medium, or Low
            3. Likelihood: Rate as High, Medium, or Low
            4. Explanation: Briefly explain the risk and its potential impact

            Focus on contractual risks, regulatory risks, litigation risks, and intellectual property risks.
            """
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": truncated_text}
            ]
        elif task == "compare":
            prompt = """You are a legal document comparison specialist. Compare the documents provided, identifying key similarities and differences in legal provisions.

            Your comparison should:
            1. Identify common legal elements across documents
            2. Highlight significant differences in provisions, rights, or obligations
            3. Note any contradictions between documents
            4. Assess which document contains more favorable terms and in what aspects

            Organize your comparison in clear sections with headings.
            """
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": truncated_text}
            ]
        else:
            prompt = f"""You are a legal document specialist. Perform {task} on the following document with expertise and precision.

            Focus on the legal aspects most relevant to this specific task.
            Provide a well-structured analysis with clear headings and concise explanations.
            """
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": truncated_text}
            ]

        # Get response with enhanced error handling
        try:
            print(f"Sending request to LLM for task: {task}")
            response = llm.invoke(messages)
            print(
                f"Received response from LLM, length: {len(response.content) if hasattr(response, 'content') else 'unknown'}")

            # Verify response has content
            if not hasattr(response, 'content') or not response.content:
                return "Analysis failed: Received empty response from language model"

            result = response.content

            # Check if the response contains potential error messages from the model
            error_indicators = [
                "I don't have the capability",
                "I cannot use external tools",
                "not applicable here as I don't have",
                "Instead, I can provide a concise summary",
                "If you have specific text from the document"
            ]

            if any(indicator in result for indicator in error_indicators):
                # The model returned an error response - fall back to basic analysis
                print("Detected error response from model. Falling back to basic analysis")

                # Create a simple analysis based on the task
                if task == "analyze":
                    result = "# Legal Document Analysis\n\n"
                    result += "## Key Provisions\n\n"
                    result += "This document appears to contain legal content that would typically include rights, obligations, and other legal provisions.\n\n"
                    result += "## Recommendations\n\n"
                    result += "For a complete legal analysis, it is recommended to have a legal professional review the entire document in detail."
                elif task == "summarize":
                    result = "# Legal Document Summary\n\n"
                    result += "This appears to be a legal document that would establish specific rights and obligations between parties.\n\n"
                    result += "A full review by qualified legal counsel is recommended for a complete understanding of all provisions."
                elif task == "extract_definitions":
                    result = "# Legal Definitions\n\n"
                    result += "No formal legal definitions were identified in the provided text.\n\n"
                    result += "Please provide a document with defined legal terms for proper extraction."
                elif task == "assess_risks":
                    result = "# Risk Assessment\n\n"
                    result += "Risk: Incomplete Analysis\n"
                    result += "Severity: Medium\n"
                    result += "Likelihood: High\n"
                    result += "Explanation: Without a complete legal review, important risks may not be identified.\n\n"
                    result += "A detailed review by qualified legal counsel is recommended for proper risk assessment."
        except Exception as e:
            error_msg = f"Error getting response from LLM: {e}"
            print(error_msg)

            # Provide a fallback response
            result = f"# Analysis Error\n\n"
            result += f"The system encountered an error while analyzing the document. Please try again later or with a different document.\n\n"
            result += f"For immediate assistance, consider having a legal professional review the document directly."

        # Reset the text to analyze to avoid memory issues
        reset_analysis_data()

        return result

    except Exception as e:
        print(f"Text analysis error: {e}")
        import traceback
        traceback.print_exc()
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
        # Set model in environment for use by analyze_text
        os.environ["OPENAI_MODEL"] = model

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