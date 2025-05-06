import os
from typing import Dict, List, Any
import chromadb
from langchain_openai import ChatOpenAI


class ChromaDBRetrievalTool:
    """Tool for retrieving documents from ChromaDB."""

    def __init__(self, db_path: str = "/home/cdsw/02_application/chromadb", collection_name: str = "document_chunks"):
        """Initialize the ChromaDB retrieval tool."""
        self.name = "ChromaDBRetrievalTool"
        self.description = "Retrieves documents from ChromaDB based on queries"
        self.db_path = db_path
        self.collection_name = collection_name

        # Will be initialized on first use
        self._client = None
        self._collection = None

    def _initialize(self):
        """Initialize ChromaDB client and collection."""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.db_path)
            try:
                self._collection = self._client.get_collection(name=self.collection_name)
                print(f"Retrieved existing collection: {self.collection_name}")
            except:
                # Use embedding function similar to main ChromaDBStorage
                try:
                    from chromadb.utils import embedding_functions
                    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2"
                    )
                except:
                    print("Using default embedding function")
                    from chromadb.utils import embedding_functions
                    ef = embedding_functions.DefaultEmbeddingFunction()

                print(f"Collection {self.collection_name} not found, creating it")
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    embedding_function=ef
                )

    def __call__(self, query: str, n_results: int = 5) -> str:
        """Run the tool to retrieve documents from ChromaDB."""
        try:
            self._initialize()
            results = self._collection.query(
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
            print(f"ChromaDBRetrievalTool error: {e}")
            return f"Error retrieving documents: {str(e)}"


class OllamaAnalysisTool:
    """Tool for analyzing text using OpenAI (renamed from Ollama for compatibility with YAML)."""

    def __init__(self, max_length: int = 5000, model: str = "gpt-4o"):
        """Initialize the analysis tool."""
        self.name = "OllamaAnalysisTool"
        self.description = "Analyzes text using OpenAI LLM"
        self.max_length = max_length
        self.model = model
        self.text_to_analyze = None
        self.current_task = None

    def __call__(self, query: str = None) -> str:
        """Run the analysis tool."""
        # Get text from instance attribute if available
        text = self.text_to_analyze or query
        task = self.current_task or "analyze"

        if not text:
            return "No text provided for analysis"

        # Truncate text if needed
        truncated_text = text[:self.max_length] if len(text) > self.max_length else text

        try:
            # Get API key from environment
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No OpenAI API key found in environment")

            # Initialize LLM
            llm = ChatOpenAI(
                api_key=api_key,
                model=self.model,
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
            self.text_to_analyze = None
            self.current_task = None
            return response.content

        except Exception as e:
            print(f"OllamaAnalysisTool error: {e}")
            # Reset the text to analyze to avoid memory issues
            self.text_to_analyze = None
            self.current_task = None
            return f"Analysis failed: {str(e)}"