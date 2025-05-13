# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. ("Cloudera") to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

import os
import yaml
import json
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import docx
import PyPDF2
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crew_tools import (
    ChromaDBStorage, get_storage, add_texts, add_summary,
    search_chunks, search_summaries, get_all_summaries, get_summary,
    ChromaDBRetrievalTool, analyze_text  # Fixed: Added analyze_text import
)
# Add these imports for the SentenceTransformer fix
import os
import sys
import logging

# Define fixed paths for YAML files - HARDCODED FOR RELIABILITY
AGENTS_YAML_PATH = "/home/cdsw/02_application/agents.yaml"
TASKS_YAML_PATH = "/home/cdsw/02_application/tasks.yaml"

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variable for SentenceTransformer cache before any imports
# that might use it
os.environ["TRANSFORMERS_CACHE"] = "/home/cdsw/02_application/model_cache"
os.makedirs("/home/cdsw/02_application/model_cache", exist_ok=True)

# Ensure all necessary directories exist
for path in ["/home/cdsw/02_application",
             "/home/cdsw/02_application/chromadb",
             "/home/cdsw/02_application/contracts",
             "/home/cdsw/02_application/results",
             "/home/cdsw/02_application/model_cache"]:
    os.makedirs(path, exist_ok=True)
    logger.info(f"Ensured directory exists: {path}")

# Try to download the model in advance
try:
    logger.info("Attempting to pre-download SentenceTransformer model")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Successfully pre-downloaded SentenceTransformer model")
except Exception as e:
    logger.warning(f"Could not pre-download SentenceTransformer model: {e}")
    logger.warning("Will fall back to DefaultEmbeddingFunction if needed")


class ChromaDBStorage:
    """Storage for document chunks and summaries in ChromaDB."""

    def __init__(self, db_path: str = "/home/cdsw/02_application/chromadb", chunks_collection: str = "document_chunks",
                 summaries_collection: str = "document_summaries"):
        """Initialize ChromaDB storage."""
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
        try:
            print("Initializing ChromaDB client")
            self.client = chromadb.PersistentClient(path=db_path)
            print("ChromaDB client initialized")
        except Exception as e:
            print(f"Error initializing ChromaDB client: {e}")
            raise

        # Create embedding function - THIS MUST COME BEFORE CREATING COLLECTIONS
        try:
            from chromadb.utils import embedding_functions
            print("Creating embedding function")
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            print("Embedding function created (using SentenceTransformers)")
        except Exception as e:
            print(f"Warning: Failed to load SentenceTransformer: {e}")
            print("Falling back to DefaultEmbeddingFunction")
            # Fallback to default embedding function if SentenceTransformer fails
            from chromadb.utils import embedding_functions  # Fixed this line
            self.ef = embedding_functions.DefaultEmbeddingFunction()
            print("Default embedding function created")

        # Get or create chunks collection
        try:
            print(f"Getting chunks collection: {chunks_collection}")
            try:
                self.chunks_collection = self.client.get_collection(
                    name=chunks_collection,
                    embedding_function=self.ef
                )
                print(f"Retrieved existing chunks collection: {chunks_collection}")
            except Exception as collection_get_error:
                print(f"Collection not found, creating: {chunks_collection}")
                self.chunks_collection = self.client.create_collection(
                    name=chunks_collection,
                    embedding_function=self.ef
                )
                print(f"Created new chunks collection: {chunks_collection}")
        except Exception as e:
            print(f"Error with chunks collection: {e}")
            raise

        # Get or create summaries collection
        try:
            print(f"Getting summaries collection: {summaries_collection}")
            try:
                self.summaries_collection = self.client.get_collection(
                    name=summaries_collection,
                    embedding_function=self.ef
                )
                print(f"Retrieved existing summaries collection: {summaries_collection}")
            except Exception as collection_get_error:
                print(f"Collection not found, creating: {summaries_collection}")
                self.summaries_collection = self.client.create_collection(
                    name=summaries_collection,
                    embedding_function=self.ef
                )
                print(f"Created new summaries collection: {summaries_collection}")
        except Exception as e:
            print(f"Error with summaries collection: {e}")
            raise

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Add texts to the chunks collection."""
        print(f"Adding {len(texts)} chunks to ChromaDB collection {self.chunks_collection_name}")
        print(f"First chunk preview: {texts[0][:100]}..." if texts else "No chunks to add")
        print(f"First metadata: {metadatas[0]}" if metadatas else "No metadata")
        print(f"First ID: {ids[0]}" if ids else "No IDs")

        try:
            self.chunks_collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully added {len(texts)} chunks to ChromaDB")

            # Verify addition
            result = self.chunks_collection.get()
            print(f"Chunks collection now contains {len(result['ids'])} items")

            return f"Added {len(texts)} chunks to ChromaDB collection {self.chunks_collection_name}"
        except Exception as e:
            print(f"Error adding texts to ChromaDB: {e}")
            raise

    def add_summary(self, doc_id: str, source: str, filename: str,
                    summary: str, analysis: str):
        """Add document summary to the summaries collection."""
        print(f"Adding summary for legal document {doc_id} to ChromaDB")

        # Create a combined document with summary and analysis
        combined_text = f"SUMMARY:\n{summary}\n\nANALYSIS:\n{analysis}"

        # Create metadata
        metadata = {
            "doc_id": doc_id,
            "source": source,
            "filename": filename,
            "type": "legal_summary_analysis"
        }

        # Create ID for the summary
        summary_id = f"{doc_id}_summary"

        try:
            self.summaries_collection.add(
                documents=[combined_text],
                metadatas=[metadata],
                ids=[summary_id]
            )
            print(f"Successfully added legal summary for document {doc_id} to ChromaDB")

            # Verify addition
            result = self.summaries_collection.get()
            print(f"Summaries collection now contains {len(result['ids'])} items")

            return f"Added legal summary for document {doc_id} to ChromaDB collection {self.summaries_collection_name}"
        except Exception as e:
            print(f"Error adding legal summary to ChromaDB: {e}")
            raise

    def search_chunks(self, query: str, n_results: int = 5):
        """Search for similar texts in chunks collection."""
        print(f"Searching for chunks with query: {query}")
        results = self.chunks_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        print(f"Found {len(results['ids'][0])} matching chunks")
        return results

    def search_summaries(self, query: str, n_results: int = 5):
        """Search for similar summaries."""
        print(f"Searching for legal summaries with query: {query}")
        results = self.summaries_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        print(f"Found {len(results['ids'][0])} matching legal summaries")
        return results

    def get_all_summaries(self):
        """Get all document summaries."""
        print(f"Retrieving all legal document summaries")
        results = self.summaries_collection.get()
        print(f"Retrieved {len(results['ids'])} legal summaries")
        return results

    def get_summary(self, doc_id: str):
        """Get summary for a specific legal document."""
        print(f"Retrieving legal summary for document {doc_id}")
        summary_id = f"{doc_id}_summary"
        try:
            result = self.summaries_collection.get(ids=[summary_id])
            if result["ids"]:
                print(f"Found legal summary for document {doc_id}")
                return {
                    "doc_id": doc_id,
                    "summary_id": summary_id,
                    "text": result["documents"][0],
                    "metadata": result["metadatas"][0]
                }
            else:
                print(f"No legal summary found for document {doc_id}")
                return None
        except Exception as e:
            print(f"Error retrieving legal summary for document {doc_id}: {e}")
            return None


class CrewAIDocumentProcessor:
    """Document processor using CrewAI for legal document analysis."""

    def __init__(self,
                 api_key: str = None,
                 base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-4o"):
        """Initialize CrewAI document processor with hardcoded YAML paths."""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.model = model

        # Use hardcoded paths for YAML files
        self.agents_yaml_path = AGENTS_YAML_PATH
        self.tasks_yaml_path = TASKS_YAML_PATH

        print(f"Using agents YAML path: {self.agents_yaml_path}")
        print(f"Using tasks YAML path: {self.tasks_yaml_path}")

        # Check and verify YAML files exist
        if not os.path.exists(self.agents_yaml_path):
            print(f"WARNING: Agents YAML file not found at {self.agents_yaml_path}")
        if not os.path.exists(self.tasks_yaml_path):
            print(f"WARNING: Tasks YAML file not found at {self.tasks_yaml_path}")

        # Load agents and tasks configurations
        self.agents_config = self._load_yaml_file(self.agents_yaml_path)
        self.tasks_config = self._load_yaml_file(self.tasks_yaml_path)

        # Initialize LLM
        self._initialize_llm()

        # Initialize CrewAI agents (lazy loading - will be created when needed)
        self.agents = {}
        self.tasks = {}

        # Initialize text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

    def _load_yaml_file(self, file_path: str) -> Dict[str, Any]:
        """Load YAML configuration file with robust error handling."""
        # Always use absolute path
        abs_path = os.path.abspath(file_path)
        print(f"Loading YAML file: {abs_path}")

        if not os.path.exists(abs_path):
            print(f"YAML file not found at: {abs_path}")
            return {}

        try:
            with open(abs_path, 'r') as file:
                data = yaml.safe_load(file)
            print(f"Successfully loaded YAML file: {abs_path}")
            return data
        except Exception as e:
            print(f"Error loading YAML file {abs_path}: {e}")
            return {}

    def _initialize_llm(self):
        """Initialize language model for CrewAI."""
        try:
            # Make sure we have an API key
            if not self.api_key:
                raise ValueError("API key is required")

            # Clean the base URL to avoid path duplication
            base_url = self.base_url
            if base_url.endswith("/chat/completions"):
                base_url = base_url.replace("/chat/completions", "")
                print(f"Removed '/chat/completions' from base_url to avoid duplication")

            # Also check for v1/chat/completions pattern
            if "/v1/chat/completions" in base_url:
                base_url = base_url.replace("/chat/completions", "")
                print(f"Removed '/chat/completions' from base_url to avoid duplication")

            # Use CrewAI's LLM class directly with the documented approach
            from crewai import LLM

            # Make sure model starts with "openai/"
            model_name = self.model
            if not model_name.startswith("openai/"):
                model_name = f"openai/{model_name}"
                print(f"Added openai/ prefix to model name: {model_name}")

            # Create the LLM with proper format
            self.llm = LLM(
                model=model_name,
                base_url=base_url,  # Use the cleaned base_url
                api_key=self.api_key,
                temperature=0.2
            )

            # Also create a direct OpenAI client for use in methods that bypass CrewAI
            from openai import OpenAI
            self.openai_client = OpenAI(
                base_url=base_url,  # Use the cleaned base_url
                api_key=self.api_key,
            )

            print(f"Initialized LLM: {model_name} at {base_url}")
        except Exception as e:
            error_msg = f"Error initializing LLM: {e}"
            print(error_msg)
            self.llm = None
            raise ValueError(f"Failed to initialize LLM: {error_msg}")

    def _create_agent(self, agent_type: str) -> Agent:
        """Create a CrewAI agent from configuration."""
        if agent_type not in self.agents_config:
            # Create a default agent if not found in config
            print(f"Agent type '{agent_type}' not found in configuration, using default")
            role = "Legal Document Processor"
            goal = "Process legal documents efficiently and extract key information"
            backstory = "An expert in legal document analysis with experience in processing legal documents"
            tools = []
        else:
            # Get agent config
            agent_config = self.agents_config[agent_type]
            role = agent_config.get("role", "Legal Document Assistant")
            goal = agent_config.get("goal", "Process legal documents efficiently")
            backstory = agent_config.get("backstory", "An expert in legal document analysis")

            # Initialize tools if specified
            tools = []
            if "tools" in agent_config:
                for tool_config in agent_config["tools"]:
                    tool_name = tool_config.get("name")
                    tool_args = tool_config.get("args", {})

                    # Create the appropriate tool
                    if tool_name == "ChromaDBRetrievalTool":
                        # Use the search_chunks function directly
                        tools.append(search_chunks)
                    elif tool_name == "OllamaAnalysisTool":
                        # Use the analyze_text function directly
                        tools.append(analyze_text)

        # Create agent
        try:
            agent = Agent(
                role=role,
                goal=goal,
                backstory=backstory,
                llm=self.llm,
                verbose=True,
                allow_delegation=True,
                tools=tools  # Pass the tools to the agent
            )
            return agent
        except Exception as e:
            print(f"Error creating agent: {e}")
            raise ValueError(f"Failed to create agent {agent_type}: {e}")

    def _create_task(self, task_type: str, context: Dict[str, Any] = None, **kwargs) -> Task:
        """Create a CrewAI task from configuration with context."""
        if task_type not in self.tasks_config:
            raise ValueError(f"Task type '{task_type}' not found in configuration")

        task_config = self.tasks_config[task_type]

        # Get agent type
        agent_type = task_config.get("agent", "document_processor")

        # Get or create agent
        if agent_type not in self.agents:
            self.agents[agent_type] = self._create_agent(agent_type)

        # Format the description with kwargs values
        description = task_config.get("description", "")
        try:
            # Format with kwargs, handling missing keys gracefully
            for key, value in kwargs.items():
                if f"{{{key}}}" in description:
                    description = description.replace(f"{{{key}}}", str(value))

            # If the description contains {doc_text}, replace it with a note about using the tool
            if "{doc_text}" in description:
                description = description.replace("{doc_text}",
                                                  "[Use the analysis tool to process the provided document]")
        except Exception as e:
            print(f"Warning: Error formatting task description: {e}")

        # Prepare context list (CrewAI expects a list for context)
        context_list = []
        if context:
            if isinstance(context, dict):
                for key, value in context.items():
                    context_list.append(f"{key}: {value}")
            elif isinstance(context, list):
                context_list = context

        # First try with context
        try:
            task = Task(
                description=description,
                expected_output=task_config.get("expected_output", ""),
                agent=self.agents[agent_type],
                context=context_list
            )
            return task
        except Exception as e:
            print(f"Error creating task with context: {e}")
            # Try without context
            try:
                task = Task(
                    description=description,
                    expected_output=task_config.get("expected_output", ""),
                    agent=self.agents[agent_type]
                )
                return task
            except Exception as e2:
                print(f"Error creating task without context: {e2}")
                raise ValueError(f"Failed to create task {task_type}: {e2}")

    def _set_tool_data(self, agent, text, task=None):
        """Set text data for analysis."""
        if not agent or not text:
            return

        # Use the global helper function
        from crew_tools import set_text_for_analysis
        set_text_for_analysis(text, task)
        print(f"Set text for analysis, task: {task}")

    def _reset_tool_data(self, agent):
        """Reset text data for analysis."""
        if not agent:
            return

        # Use the global helper function
        from crew_tools import reset_analysis_data
        reset_analysis_data()
        print("Reset analysis data")

    def extract_text(self, file_path: str) -> str:
        """Extract text from various document formats."""
        print(f"Extracting text from document: {file_path}")
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_extension == '.docx':
            return self._extract_from_docx(file_path)
        elif file_extension == '.txt':
            return self._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF files with improved reliability for legal documents."""
        print(f"Extracting text from PDF: {file_path}")

        # Try multiple extraction methods until we get usable text
        text = ""

        # Method 1: PyPDF2 with enhanced handling for legal documents
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""

                # Process each page individually to mitigate failures
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"  # Double newline for paragraph separation
                    except Exception as e:
                        print(f"Warning: Error extracting page {page_num}: {e}")
                        # Continue with next page if one fails

                print(f"PyPDF2 extracted {len(text)} characters from PDF")

                # If we got a reasonable amount of text, clean and return it
                if len(text) > 100:
                    text = self._clean_pdf_artifacts(text, file_path)
                    return text
        except Exception as e:
            print(f"Error extracting text with PyPDF2: {e}")

        # If PyPDF2 didn't work well, try another method using pdftotext if available
        try:
            import subprocess
            import tempfile

            # Create a temporary file for the output
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
                temp_output = temp_file.name

            # Try with layout preservation first
            try:
                subprocess.run(['pdftotext', '-layout', file_path, temp_output],
                               check=True, capture_output=True, timeout=60)

                # Read the output
                with open(temp_output, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()

                print(f"pdftotext with layout extracted {len(text)} characters from PDF")

                # If not enough text, try without layout
                if len(text) < 100:
                    subprocess.run(['pdftotext', file_path, temp_output],
                                   check=True, capture_output=True, timeout=30)

                    with open(temp_output, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()

                    print(f"pdftotext without layout extracted {len(text)} characters from PDF")

                # Clean up
                os.unlink(temp_output)

                # If we got a reasonable amount of text, clean and return it
                if len(text) > 100:
                    text = self._clean_pdf_artifacts(text, file_path)
                    return text
            except subprocess.CalledProcessError:
                print("pdftotext command failed")
            except FileNotFoundError:
                print("pdftotext command not found")
            except Exception as e:
                print(f"pdftotext error: {e}")
            finally:
                # Ensure temp file is removed
                if os.path.exists(temp_output):
                    try:
                        os.unlink(temp_output)
                    except:
                        pass
        except Exception as e:
            print(f"Error using alternative PDF extraction: {e}")

        # As a last resort, try raw binary reading with different encodings
        try:
            with open(file_path, 'rb') as file:
                binary_data = file.read()
                best_text = ""
                best_score = 0

                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252', 'utf-16', 'ascii']:
                    try:
                        text = binary_data.decode(encoding, errors='ignore')

                        # For legal documents, look for common legal terms
                        legal_term_count = sum(1 for term in [
                            'agreement', 'hereby', 'shall', 'party', 'contract',
                            'obligations', 'pursuant', 'herein', 'witness', 'whereas'
                        ] if term.lower() in text.lower())

                        # Consider both length and presence of legal terms
                        quality_score = len(text) * (1 + (legal_term_count * 0.1))

                        print(f"Raw binary with {encoding} extracted {len(text)} characters (score: {quality_score})")

                        if quality_score > best_score:
                            best_text = text
                            best_score = quality_score
                    except Exception:
                        pass

                if best_score > 0:
                    text = best_text
                    # If we got text that looks reasonable, clean and return it
                    if len(text) > 100:
                        text = self._clean_pdf_artifacts(text, file_path)
                        return text
        except Exception as e:
            print(f"Error with raw binary reading: {e}")

        # If we got here, we couldn't extract good text
        # Return whatever we have, even if it's not great
        if not text:
            text = f"Failed to extract text from PDF: {file_path}"
            print(text)
        else:
            text = self._clean_pdf_artifacts(text, file_path)

        return text

    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX files."""
        print(f"Extracting text from DOCX: {file_path}")
        try:
            doc = docx.Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            text = '\n'.join(full_text)
            print(f"Extracted {len(text)} characters from DOCX")
            return text
        except Exception as e:
            print(f"Error extracting text from DOCX: {e}")
            raise

    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT files."""
        print(f"Extracting text from TXT: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            print(f"Extracted {len(text)} characters from TXT")
            return text
        except Exception as e:
            print(f"Error extracting text from TXT: {e}")
            raise

    def process_document(self, file_path: str, doc_id: str) -> Dict[str, Any]:
        """Process a document using CrewAI with improved PDF handling."""
        print(f"Processing document: {file_path}, ID: {doc_id}")

        # Check if file exists and has content
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if os.path.getsize(file_path) == 0:
            raise ValueError(f"File is empty: {file_path}")

        # Extract text with improved PDF handling
        text = self.extract_text(file_path)

        # Verify we have text
        if not text or len(text) < 10:
            print(f"Warning: Very little text extracted from {file_path}. File may be corrupted or unreadable.")
            text = f"[Unable to extract meaningful text from: {file_path}]"

        # Log information about the extracted text
        print(f"Extracted {len(text)} characters from document")
        print(f"Text preview: {text[:200]}...")

        # Clean the text of potential PDF artifacts
        text = self._clean_pdf_artifacts(text, file_path)
        print(f"After cleaning, text length: {len(text)} characters")

        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        print(f"Split text into {len(chunks)} chunks")

        # Prepare metadata
        metadatas = [{"source": file_path, "doc_id": doc_id, "chunk": i} for i in range(len(chunks))]

        # Prepare IDs
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

        # Create task for document processing using the tasks.yaml
        try:
            # Get or create the document processor agent
            if "document_processor" not in self.agents:
                self.agents["document_processor"] = self._create_agent("document_processor")

            # Set text in analysis tool if available
            self._set_tool_data(self.agents["document_processor"], text[:5000], "process")

            # Create a description that doesn't include the full text
            description = f"Process the document with ID {doc_id} and prepare it for analysis."

            # Create the task
            process_task = Task(
                description=description,
                expected_output="Processed document metadata",
                agent=self.agents["document_processor"]
            )

            # Create crew for processing
            crew = Crew(
                agents=[self.agents["document_processor"]],
                tasks=[process_task],
                verbose=True,
                process=Process.sequential
            )

            # Execute the crew
            crew_result = crew.kickoff()
            print(f"CrewAI processing result: {crew_result}")

            # Reset tool data
            self._reset_tool_data(self.agents["document_processor"])

        except Exception as e:
            print(f"Warning: Error in CrewAI processing: {e}")
            import traceback
            traceback.print_exc()
            # Continue with basic processing even if CrewAI fails
            crew_result = f"Document processed without CrewAI: {e}"

        return {
            "text": text,
            "chunks": chunks,
            "metadatas": metadatas,
            "ids": ids,
            "doc_id": doc_id,
            "crew_result": crew_result
        }

    def _clean_pdf_artifacts(self, text: str, file_path: str = None) -> str:
        """
        Clean PDF artifacts from extracted text.
        Specifically enhanced for legal documents like farmout agreements.
        """
        # Remove PDF object references
        text = re.sub(r'\d+ \d+ obj.*?endobj', ' ', text, flags=re.DOTALL)

        # Remove PDF streams
        text = re.sub(r'stream\s.*?endstream', ' ', text, flags=re.DOTALL)

        # Remove PDF dictionary objects
        text = re.sub(r'<<.*?>>', ' ', text, flags=re.DOTALL)

        # Remove PDF operators and commands
        text = re.sub(r'/[A-Za-z0-9_]+\s+', ' ', text)

        # Remove PDF metadata like CID, CMap entries
        text = re.sub(r'/(CID|CMap|Registry|Ordering|Supplement|CIDToGIDMap).*?def', ' ', text)

        # Remove "R" references
        text = re.sub(r'\d+ \d+ R', ' ', text)

        # Remove common PDF artifacts
        text = re.sub(r'EvoPdf_[a-zA-Z0-9_]+', '', text)

        # Remove URLs that appear in farmout PDFs (common in SEC documents)
        text = re.sub(r'https?://www\.sec\.gov/[^\s]+', '', text)

        # Remove timestamp headers that may appear in SEC documents
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}\s+[AP]M', '', text)

        # Remove page numbers in common formats
        text = re.sub(r'\n\s*\d+\s*/\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*Page\s+\d+\s+of\s+\d+\s*\n', '\n', text)

        # Fix paragraph breaks (especially for farmout agreements)
        text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)  # Join broken sentences

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Restore paragraph breaks for numbered sections (common in farmout agreements)
        text = re.sub(r'(\d+\.\s+)', r'\n\n\1', text)

        # Restore paragraph breaks for WHEREAS clauses
        text = re.sub(r'(WHEREAS)', r'\n\n\1', text)

        # Normalize whitespace again and trim
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def summarize(self, text: str, max_length: int = 2500) -> str:
        """Summarize text using CrewAI with robust fallback."""
        print(f"Summarizing text, length: {len(text)} characters")

        # Limit text to avoid token limits
        truncated_text = text[:max_length * 2] if len(text) > max_length * 2 else text

        try:
            # First attempt: Use the CrewAI approach
            try:
                # Load task configuration
                if "summarize_document" not in self.tasks_config:
                    print("Task 'summarize_document' not found in configuration, using fallback")
                    raise ValueError("Task configuration not found")

                task_config = self.tasks_config["summarize_document"]

                # Get agent type
                agent_type = task_config.get("agent", "document_summarizer")

                # Get or create agent
                if agent_type not in self.agents:
                    self.agents[agent_type] = self._create_agent(agent_type)

                # Set the text in the tool instead of in the description
                self._set_tool_data(self.agents[agent_type], truncated_text, "summarize")

                # Create modified description WITHOUT the doc_text
                description = task_config.get("description", "")
                # Remove {doc_text} placeholder and replace with instruction to use the tool
                description = description.replace("{doc_text}",
                                                  "[Use the OllamaAnalysisTool to analyze the provided document]")
                if "{max_length}" in description:
                    description = description.replace("{max_length}", str(max_length))

                # Create task without context
                summarize_task = Task(
                    description=description,
                    expected_output=task_config.get("expected_output", ""),
                    agent=self.agents[agent_type]
                )

                # Create crew for summarization
                crew = Crew(
                    agents=[self.agents[agent_type]],
                    tasks=[summarize_task],
                    verbose=True,
                    process=Process.sequential
                )

                # Execute the crew
                result = crew.kickoff()

                # Reset tool data to avoid memory issues
                self._reset_tool_data(self.agents[agent_type])

                # Convert CrewOutput to string
                summary_text = str(result) if hasattr(result, '__str__') else None

                print(f"CrewAI summary length: {len(summary_text) if summary_text else 0}")

                # Check if the summary looks valid
                if not summary_text or len(summary_text) < 50 or "I don't have the capability" in summary_text:
                    print("CrewAI summary appears invalid, using fallback")
                    raise ValueError("Invalid summary from CrewAI")

                return summary_text

            except Exception as e:
                print(f"CrewAI summarization failed, using fallback: {e}")
                raise e  # Re-raise to trigger fallback

        except Exception as e:
            print(f"Using fallback summarization: {e}")

            # Fallback method: Simple rule-based summary
            text_sample = truncated_text.lower()
            summary = ""

            # Detect document type
            doc_type = "legal document"
            if "agreement" in text_sample:
                doc_type = "agreement"
            elif "contract" in text_sample:
                doc_type = "contract"
            elif "amendment" in text_sample:
                doc_type = "amendment"
            elif "policy" in text_sample:
                doc_type = "policy"
            elif "memorandum" in text_sample:
                doc_type = "memorandum"

            # Create a simple summary introduction
            summary = f"This document appears to be a {doc_type} that establishes legal rights and obligations.\n\n"

            # Key provisions
            summary += "The document likely addresses:\n\n"
            provisions = []

            if "term" in text_sample or "duration" in text_sample:
                provisions.append("Term/Duration")

            if "payment" in text_sample or "compensation" in text_sample:
                provisions.append("Payment Terms")

            if "terminat" in text_sample:
                provisions.append("Termination Provisions")

            if "confidential" in text_sample:
                provisions.append("Confidentiality")

            if "intellectual property" in text_sample or "copyright" in text_sample:
                provisions.append("Intellectual Property")

            if provisions:
                for provision in provisions:
                    summary += f"- {provision}\n"
            else:
                summary += "- Various legal terms and conditions\n"

            summary += "\nNote: This is an automated summary generated by the fallback system. For a comprehensive understanding, please consult a qualified legal professional."

            return summary

    def analyze(self, text: str, analysis_depth: str = "detailed") -> str:
        """Analyze text directly without using LLM."""
        print(f"Analyzing text, length: {len(text)} characters, depth: {analysis_depth}")

        # Truncate text if needed
        max_length = 15000
        truncated_text = text[:max_length] if len(text) > max_length else text
        text_sample = truncated_text.lower()

        # Create a rule-based analysis
        analysis = "# Legal Document Analysis\n\n"

        # Detect document type
        if "agreement" in text_sample:
            doc_type = "Agreement"
        elif "contract" in text_sample:
            doc_type = "Contract"
        elif "amendment" in text_sample:
            doc_type = "Amendment"
        elif "policy" in text_sample:
            doc_type = "Policy"
        elif "memorandum" in text_sample:
            doc_type = "Memorandum"
        else:
            doc_type = "Legal Document"

        analysis += f"## Document Type\n\nThis document appears to be a {doc_type}.\n\n"

        # Key provisions
        analysis += "## Key Provisions\n\n"
        provisions = []

        if "term" in text_sample or "duration" in text_sample:
            provisions.append("Term and Duration: The document specifies the time period for which it is valid.")

        if "payment" in text_sample or "compensation" in text_sample or "fee" in text_sample:
            provisions.append("Payment Terms: The document outlines payment obligations, schedules, or compensation.")

        if "terminat" in text_sample:
            provisions.append("Termination Provisions: Conditions under which the agreement may be terminated.")

        if "confidential" in text_sample:
            provisions.append("Confidentiality: Requirements to maintain the confidentiality of certain information.")

        if "intellectual property" in text_sample or "copyright" in text_sample or "patent" in text_sample:
            provisions.append(
                "Intellectual Property: Provisions regarding ownership and rights to intellectual property.")

        if "indemn" in text_sample:
            provisions.append("Indemnification: Provisions for protection against certain losses or damages.")

        if "warranty" in text_sample or "warranties" in text_sample:
            provisions.append("Warranties: Assurances or guarantees provided by one party to another.")

        if "represent" in text_sample:
            provisions.append("Representations: Statements of fact made by the parties.")

        if "govern" in text_sample and ("law" in text_sample or "jurisdiction" in text_sample):
            provisions.append("Governing Law: Specifies which jurisdiction's laws govern the document.")

        if "dispute" in text_sample or "arbitration" in text_sample or "mediation" in text_sample:
            provisions.append("Dispute Resolution: Procedures for resolving disagreements between parties.")

        if provisions:
            for provision in provisions:
                analysis += f"- {provision}\n"
        else:
            analysis += "No clear provisions were identified in the sample text examined.\n"

        # Obligations
        analysis += "\n## Obligations\n\n"

        obligations = []

        # Look for obligation indicators
        obligation_sentences = []
        sentences = re.split(r'[.!?]+', truncated_text)

        for sentence in sentences:
            lower_sentence = sentence.lower().strip()
            if any(term in lower_sentence for term in [
                " shall ", "must", "is obligated", "is required", "agrees to", "duty to",
                "responsible for", "obligation", "required to"
            ]):
                obligation_sentences.append(sentence.strip())

        # Extract up to 5 obligation sentences
        selected_obligations = obligation_sentences[:5] if obligation_sentences else []

        if selected_obligations:
            analysis += "The document contains obligations including:\n\n"
            for i, obligation in enumerate(selected_obligations, 1):
                analysis += f"{i}. {obligation}\n\n"
        else:
            analysis += "No specific obligations were identified in the sample text examined.\n\n"

        # Rights
        analysis += "## Rights\n\n"

        rights = []

        # Look for rights indicators
        rights_sentences = []

        for sentence in sentences:
            lower_sentence = sentence.lower().strip()
            if any(term in lower_sentence for term in [
                " may ", "entitle", "right to", "is permitted", "allowed to",
                "authority to", "option to", "discretion"
            ]) and not any(term in lower_sentence for term in ["shall not", "may not", "not permitted", "not allowed"]):
                rights_sentences.append(sentence.strip())

        # Extract up to 5 rights sentences
        selected_rights = rights_sentences[:5] if rights_sentences else []

        if selected_rights:
            analysis += "The document grants rights including:\n\n"
            for i, right in enumerate(selected_rights, 1):
                analysis += f"{i}. {right}\n\n"
        else:
            analysis += "No specific rights were identified in the sample text examined.\n\n"

        # Important clauses
        analysis += "## Important Clauses\n\n"

        important_clauses = []

        # Important clause indicators
        important_terms = [
            "notwithstanding", "subject to", "provided that", "except as",
            "without limitation", "in the event", "for the avoidance of doubt",
            "material breach", "force majeure", "assignment", "amendments", "waiver"
        ]

        clause_sentences = []

        for sentence in sentences:
            lower_sentence = sentence.lower().strip()
            if any(term in lower_sentence for term in important_terms):
                clause_sentences.append(sentence.strip())

        # Extract up to 5 important clauses
        selected_clauses = clause_sentences[:5] if clause_sentences else []

        if selected_clauses:
            analysis += "Important clauses in the document include:\n\n"
            for i, clause in enumerate(selected_clauses, 1):
                analysis += f"{i}. {clause}\n\n"
        else:
            analysis += "No specific important clauses were identified in the sample text examined.\n\n"

        # Add disclaimer
        analysis += "## Disclaimer\n\n"
        analysis += "This is an automated analysis and may not capture all legal nuances. For a comprehensive analysis, please consult a qualified legal professional."

        return analysis

    def compare_with_summaries(self, new_text: str, summaries: List[Dict[str, Any]],
                               focus_areas: List[str] = None) -> str:
        """Compare a new document with existing summaries using direct API call."""
        print(f"Comparing new document with {len(summaries)} existing summaries")

        if not focus_areas:
            focus_areas = ["legal provisions", "obligations", "rights", "liability", "termination", "jurisdiction"]

        # Limit text to avoid token limits
        truncated_new_text = new_text[:3000] if len(new_text) > 3000 else new_text

        try:
            # Prepare summaries text
            summaries_text = "\n\n".join(
                [f"DOCUMENT {i + 1} (ID: {summary.get('doc_id', f'doc_{i + 1}')}): {summary.get('text', '')}"
                 for i, summary in enumerate(summaries)])

            # Limit summaries text length
            truncated_summaries_text = summaries_text[:5000] if len(summaries_text) > 5000 else summaries_text

            # Try direct OpenAI call first
            try:
                # Extract model name without provider prefix
                model_name = self.model
                if "/" in model_name:
                    parts = model_name.split("/")
                    if len(parts) > 1:
                        # If format is "openai/meta/llama-3.1", use the last two parts
                        if len(parts) > 2:
                            model_name = "/".join(parts[1:])
                        else:
                            model_name = parts[1]

                # Create prompt for comparison
                prompt = f"""
                Compare the NEW DOCUMENT with the EXISTING DOCUMENTS, focusing on: {', '.join(focus_areas)}.

                Identify key similarities and differences in legal terms, obligations, and rights.
                Provide a well-structured analysis that highlights important legal distinctions.

                NEW DOCUMENT:
                {truncated_new_text}

                EXISTING DOCUMENTS:
                {truncated_summaries_text}
                """

                # Call OpenAI directly
                response = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a legal document comparison specialist."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1500
                )

                comparison_text = response.choices[0].message.content
                print(f"Got direct document comparison response")

                return comparison_text
            except Exception as direct_call_error:
                print(f"Error with direct OpenAI call: {direct_call_error}")
                # Fall back to CrewAI approach

                # Get or create the document comparer agent
                if "document_comparer" not in self.agents:
                    self.agents["document_comparer"] = self._create_agent("document_comparer")

                # Set combined text in the tool
                combined_text = f"NEW DOCUMENT:\n{truncated_new_text}\n\nEXISTING DOCUMENTS:\n{truncated_summaries_text}"
                self._set_tool_data(self.agents["document_comparer"], combined_text, "compare")

                # Create description directly without using task configuration to avoid token limits
                description = "Compare the following legal documents, focusing on: " + ", ".join(focus_areas) + ". "
                description += "Identify key similarities and differences in legal terms, obligations, and rights. "
                description += "Provide a well-structured analysis that highlights important legal distinctions. "
                description += f"Document IDs: {','.join([summary.get('doc_id', f'doc_{i}') for i, summary in enumerate(summaries)])}"
                description += "\n\nUse the OllamaAnalysisTool to process the documents for comparison."

                # Create agent and task directly
                compare_task = Task(
                    description=description,
                    expected_output="A comprehensive legal comparison highlighting key similarities and differences",
                    agent=self.agents["document_comparer"]
                )

                # Create crew for comparison
                crew = Crew(
                    agents=[self.agents["document_comparer"]],
                    tasks=[compare_task],
                    verbose=True,
                    process=Process.sequential
                )

                # Execute the crew
                result = crew.kickoff()

                # Reset tool data
                self._reset_tool_data(self.agents["document_comparer"])

                # Convert CrewOutput to string
                comparison_text = str(result) if hasattr(result,
                                                         '__str__') else "Error: Unable to convert result to string"

                # Format the output if it looks like JSON
                if comparison_text.strip().startswith('{') or comparison_text.strip().startswith('['):
                    try:
                        import json
                        data = json.loads(comparison_text)

                        # Convert JSON to a more readable format
                        formatted_text = "# Document Comparison Results\n\n"

                        # Format according to expected structure
                        if isinstance(data, dict):
                            if "similarities" in data:
                                formatted_text += "## Similarities\n\n"
                                for item in data["similarities"]:
                                    formatted_text += f"- {item}\n"
                                formatted_text += "\n"

                            if "differences" in data:
                                formatted_text += "## Differences\n\n"
                                for item in data["differences"]:
                                    formatted_text += f"- {item}\n"
                                formatted_text += "\n"

                            if "analysis" in data:
                                formatted_text += "## Analysis\n\n"
                                formatted_text += data["analysis"]
                                formatted_text += "\n"

                            # Add any other sections
                            for key, value in data.items():
                                if key not in ["similarities", "differences", "analysis"]:
                                    formatted_text += f"## {key.replace('_', ' ').title()}\n\n"
                                    formatted_text += f"{value}\n\n"

                        return formatted_text
                    except:
                        # If we can't parse as JSON, return the raw string
                        pass

                return comparison_text

        except Exception as e:
            print(f"Error in compare_with_summaries method: {e}")
            import traceback
            traceback.print_exc()
            # Fallback if CrewAI fails
            return f"Error comparing documents: {str(e)}"
    def extract_legal_definitions(self, text: str) -> str:
        """Extract and analyze legal definitions from a document."""
        print(f"Extracting legal definitions, text length: {len(text)} characters")

        # Limit text to avoid token limits
        truncated_text = text[:5000] if len(text) > 5000 else text

        try:
            # Use direct OpenAI call to extract definitions
            try:
                # Extract model name without provider prefix
                model_name = self.model
                if "/" in model_name:
                    parts = model_name.split("/")
                    if len(parts) > 1:
                        # If format is "openai/meta/llama-3.1", use the last two parts
                        if len(parts) > 2:
                            model_name = "/".join(parts[1:])
                        else:
                            model_name = parts[1]

                # Create prompt for legal definitions
                prompt = """
                Extract and explain all defined legal terms from the following document text.
                Identify inconsistencies in definitions and potential legal ambiguities.

                For each term, provide:
                1. The term being defined
                2. The definition provided in the document
                3. Any potential ambiguities or issues with the definition

                Format your response clearly with the term followed by its definition.

                Document text:
                """ + truncated_text

                # Call OpenAI directly
                response = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a legal terminology specialist."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1500
                )

                definitions_text = response.choices[0].message.content
                print(f"Got direct legal definitions extraction response")

                return definitions_text
            except Exception as direct_call_error:
                print(f"Error with direct OpenAI call: {direct_call_error}")
                # Fall back to CrewAI approach

                # Get or create the legal terminology extractor agent
                if "legal_terminology_extractor" not in self.agents:
                    self.agents["legal_terminology_extractor"] = self._create_agent("legal_terminology_extractor")

                # Set the text in the analysis tool
                self._set_tool_data(self.agents["legal_terminology_extractor"], truncated_text, "extract_definitions")

                # Create description directly without using task configuration
                description = "Extract and explain all defined legal terms from the document text. "
                description += "Identify inconsistencies in definitions and potential legal ambiguities. "
                description += "Use the OllamaAnalysisTool to process the document text."

                # Create task
                extract_task = Task(
                    description=description,
                    expected_output="Glossary of legal terms with explanations and identified issues",
                    agent=self.agents["legal_terminology_extractor"]
                )

                # Create crew for extraction
                crew = Crew(
                    agents=[self.agents["legal_terminology_extractor"]],
                    tasks=[extract_task],
                    verbose=True,
                    process=Process.sequential
                )

                # Execute the crew
                result = crew.kickoff()

                # Reset tool data
                self._reset_tool_data(self.agents["legal_terminology_extractor"])

                # Convert CrewOutput to string
                return str(result) if hasattr(result, '__str__') else "Error: Unable to convert result to string"
        except Exception as e:
            print(f"Error in extract_legal_definitions method: {e}")
            import traceback
            traceback.print_exc()
            # Fallback if CrewAI fails
            return f"Error extracting legal definitions: {str(e)}"

    def extract_legal_definitions_improved(self, text: str) -> str:
        """Extract and analyze legal definitions from a document with improved accuracy."""
        # This is just a wrapper around extract_legal_definitions to match the API expected by the app
        return self.extract_legal_definitions(text)

    def assess_legal_risks(self, text: str, risk_categories: List[str] = None) -> str:
        """Assess legal risks in a document with enhanced support for farmout agreements."""
        print(f"Assessing legal risks, text length: {len(text)} characters")

        # Check if this might be a farmout agreement
        is_farmout = any(term in text.lower() for term in
                         ['farmout', 'farm-out', 'farm out', 'farmor', 'farmee', 'working interest', 'payout'])

        if not risk_categories:
            if is_farmout:
                # Use farmout-specific risk categories
                risk_categories = [
                    "drilling obligations", "earning provisions", "working interest",
                    "assignment", "operatorship", "termination"
                ]
            else:
                # Default categories for other legal documents
                risk_categories = ["contractual", "regulatory", "litigation", "intellectual property"]

        # Limit text to avoid token limits - use a larger limit for farmout agreements
        max_chars = 7500 if is_farmout else 5000
        truncated_text = text[:max_chars] if len(text) > max_chars else text

        try:
            # Create a specialized prompt for risk assessment
            if is_farmout:
                prompt = """
                Perform a thorough risk assessment of this farmout agreement.
                A farmout agreement is a contract where one company (farmor) assigns all or part of its working interest in an oil and gas lease to another company (farmee).

                For each identified risk, provide the following format:
                - Risk: [Clear name of the risk]
                - Severity: [High, Medium, or Low]
                - Likelihood: [High, Medium, or Low]
                - Explanation: [Brief explanation]

                Focus on these key risk categories:
                """
                for category in risk_categories:
                    prompt += f"- {category.title()}\n"

                prompt += "\n" + truncated_text
            else:
                # Standard prompt for other legal documents
                prompt = """
                Perform a detailed legal risk assessment of the document.

                For each identified risk, provide the following in a clearly structured format:
                - Risk: [Clear name of the risk]
                - Severity: [High, Medium, or Low]
                - Likelihood: [High, Medium, or Low]
                - Explanation: [Brief explanation]

                Focus on these risk categories:
                """
                for i, category in enumerate(risk_categories):
                    prompt += f"{i + 1}. {category.title()} Risks\n"

                prompt += "\n" + truncated_text

            # Use direct OpenAI call instead of CrewAI for risk assessment
            try:
                # Extract just the model name without the provider prefix
                model_name = self.model
                if "/" in model_name:
                    parts = model_name.split("/")
                    if len(parts) > 1:
                        # If format is "openai/meta/llama-3.1", use the last two parts
                        if len(parts) > 2:
                            model_name = "/".join(parts[1:])
                        else:
                            model_name = parts[1]

                print(f"Using model {model_name} for direct risk assessment")

                # Call OpenAI directly
                response = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a legal risk assessment expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1500
                )

                risk_text = response.choices[0].message.content
                print(f"Got direct risk assessment response, length: {len(risk_text)}")

                return risk_text
            except Exception as direct_call_error:
                print(f"Error with direct OpenAI call: {direct_call_error}")
                # Fall back to using CrewAI approach if direct call fails

                # Get or create the legal risk assessor agent
                if "legal_risk_assessor" not in self.agents:
                    self.agents["legal_risk_assessor"] = self._create_agent("legal_risk_assessor")

                # Create task
                risk_task = Task(
                    description=prompt,
                    expected_output="Structured legal risk assessment with severity and likelihood ratings",
                    agent=self.agents["legal_risk_assessor"]
                )

                # Create crew for risk assessment
                crew = Crew(
                    agents=[self.agents["legal_risk_assessor"]],
                    tasks=[risk_task],
                    verbose=True,
                    process=Process.sequential
                )

                # Execute the crew
                result = crew.kickoff()

                # Reset tool data
                try:
                    from crew_tools import reset_analysis_data
                    reset_analysis_data()
                except:
                    pass

                # Convert CrewOutput to string if needed
                risk_text = str(result) if hasattr(result, '__str__') else "Error: Unable to convert result to string"

                return risk_text
        except Exception as e:
            error_msg = f"Error assessing legal risks: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg

    def check_legal_compliance(self, text: str, regulatory_areas: List[str] = None) -> str:
        """Check document for compliance with regulations."""
        print(f"Checking legal compliance, text length: {len(text)} characters")

        if not regulatory_areas:
            regulatory_areas = ["data privacy", "consumer protection", "employment", "intellectual property"]

        # Limit text to avoid token limits
        truncated_text = text[:5000] if len(text) > 5000 else text

        try:
            # Get or create the legal compliance checker agent
            if "legal_compliance_checker" not in self.agents:
                self.agents["legal_compliance_checker"] = self._create_agent("legal_compliance_checker")

            # Set the text in the analysis tool
            self._set_tool_data(self.agents["legal_compliance_checker"], truncated_text, "check_compliance")

            # Create task
            compliance_task = self._create_task(
                "check_legal_compliance",
                context={"regulatory_areas": regulatory_areas},
                doc_text="[Use the OllamaAnalysisTool to access the document text]"
            )

            # Create crew for compliance check
            crew = Crew(
                agents=[self.agents["legal_compliance_checker"]],
                tasks=[compliance_task],
                verbose=True,
                process=Process.sequential
            )

            # Execute the crew
            result = crew.kickoff()

            # Reset tool data
            self._reset_tool_data(self.agents["legal_compliance_checker"])

            # Convert CrewOutput to string
            return str(result) if hasattr(result, '__str__') else "Error: Unable to convert result to string"
        except Exception as e:
            print(f"Error in check_legal_compliance method: {e}")
            import traceback
            traceback.print_exc()
            # Fallback if CrewAI fails
            return f"Error checking legal compliance: {str(e)}"

    def analyze_governing_law(self, text: str) -> str:
        """Analyze governing law and jurisdiction clauses."""
        print(f"Analyzing governing law, text length: {len(text)} characters")

        # Limit text to avoid token limits
        truncated_text = text[:5000] if len(text) > 5000 else text

        try:
            # Get or create the legal analyzer agent
            if "legal_analyzer" not in self.agents:
                self.agents["legal_analyzer"] = self._create_agent("legal_analyzer")

            # Set the text in the analysis tool
            self._set_tool_data(self.agents["legal_analyzer"], truncated_text, "analyze_governing_law")

            # Create task
            law_task = self._create_task(
                "identify_governing_law",
                doc_text="[Use the OllamaAnalysisTool to access the document text]"
            )

            # Create crew for governing law analysis
            crew = Crew(
                agents=[self.agents["legal_analyzer"]],
                tasks=[law_task],
                verbose=True,
                process=Process.sequential
            )

            # Execute the crew
            result = crew.kickoff()

            # Reset tool data
            self._reset_tool_data(self.agents["legal_analyzer"])

            # Convert CrewOutput to string
            return str(result) if hasattr(result, '__str__') else "Error: Unable to convert result to string"
        except Exception as e:
            print(f"Error in analyze_governing_law method: {e}")
            import traceback
            traceback.print_exc()
            # Fallback if CrewAI fails
            return f"Error analyzing governing law: {str(e)}"


def get_document_processor():
    """Get the OpenAI document processor based on settings."""
    # Check for OpenAI configuration
    config_path = "config/settings.json"
    api_key = None
    model_provider = None

    # First check environment variable
    if os.environ.get("OPENAI_API_KEY"):
        api_key = os.environ.get("OPENAI_API_KEY")
        print("Using API key from environment variable")

    if os.environ.get("LLM_PROVIDER"):
        model_provider = os.environ.get("LLM_PROVIDER")
        print(f"Using model provider from environment variable: {model_provider}")

    # Then check settings file
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                settings = json.load(f)

                # If we didn't get a key from environment, try settings
                if not api_key and settings.get("openai_api_key"):
                    api_key = settings.get("openai_api_key")
                    print("Using API key from settings file")

                # If we didn't get a provider from environment, try settings
                if not model_provider and settings.get("model_provider"):
                    model_provider = settings.get("model_provider")
                    print(f"Using model provider from settings file: {model_provider}")

                # Get endpoint and model
                endpoint = settings.get("openai_endpoint", "https://api.openai.com/v1")
                model = settings.get("openai_model", "gpt-4o")
        else:
            # Default values if no settings file
            endpoint = "https://api.openai.com/v1"
            model = "gpt-4o"
            if not model_provider:
                model_provider = "openai"
            print("No settings file found, using defaults")

        # Final check for API key
        if not api_key:
            # One more attempt to get from session state if code is running in streamlit
            try:
                import streamlit as st
                if 'openai_api_key' in st.session_state and st.session_state.openai_api_key:
                    api_key = st.session_state.openai_api_key
                    print("Using API key from session state")
                if 'model_provider' in st.session_state and not model_provider:
                    model_provider = st.session_state.model_provider
                    print(f"Using model provider from session state: {model_provider}")
            except:
                pass

        # Check if we have an API key
        if not api_key:
            raise ValueError("No API key found in settings, environment variables, or session state.")

        # Ensure we have a model provider
        if not model_provider:
            model_provider = "openai"  # Default to OpenAI
            print(f"Using default model provider: {model_provider}")

        # Adjust model name if needed for litellm
        if model_provider != "openai" and not model.startswith(f"{model_provider}/"):
            model = f"{model_provider}/{model}"
            print(f"Adjusted model name for litellm: {model}")

        # Create the processor
        print(f"Creating CrewAI processor with endpoint: {endpoint}, model: {model}, provider: {model_provider}")
        return CrewAIDocumentProcessor(
            api_key=api_key,
            base_url=endpoint,
            model=model
        )
    except Exception as e:
        print(f"Error setting up document processor: {e}")
        raise ValueError(
            f"Unable to initialize document processor: {e}. Please configure an API key in the Settings tab.")

def load_yaml_file(file_path: str) -> Dict[str, Any]:
    """Load YAML file."""
    print(f"Loading YAML file: {file_path}")

    # Try standard absolute paths first
    if not os.path.exists(file_path):
        # Try with application directory
        app_path = os.path.join("/home/cdsw/02_application", os.path.basename(file_path))
        if os.path.exists(app_path):
            file_path = app_path
            print(f"Found YAML file at: {file_path}")

    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        print(f"Successfully loaded YAML file: {file_path}")
        return data
    except Exception as e:
        print(f"Error loading YAML file {file_path}: {e}")
        return {}


def process_documents():
    """Process all legal documents in the contracts folder."""
    print("\n========== STARTING LEGAL DOCUMENT PROCESSING ==========\n")

    # Set up storage
    print("Setting up ChromaDB storage...")
    chroma_storage = ChromaDBStorage(
        db_path="/home/cdsw/02_application/chromadb",
        chunks_collection="document_chunks",
        summaries_collection="document_summaries"
    )

    # Get the appropriate document processor based on settings
    document_processor = get_document_processor()
    processor_type = document_processor.__class__.__name__
    print(f"Using document processor: {processor_type}")

    # Process all documents in the contracts folder
    # Try both absolute and relative paths to find contracts
    possible_contract_paths = [
        "/home/cdsw/02_application/contracts",
        "contracts",
        os.path.join(os.getcwd(), "contracts"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "contracts")
    ]

    contracts_folder = None
    for path in possible_contract_paths:
        if os.path.exists(path):
            contracts_folder = path
            print(f"Found contracts folder at: {os.path.abspath(path)}")
            break

    if not contracts_folder:
        print("Contracts folder not found in any of the expected locations:")
        for path in possible_contract_paths:
            print(
                f"  - {os.path.abspath(path) if os.path.isabs(path) else os.path.abspath(os.path.join(os.getcwd(), path))}")

        # Default to the standard path and create it
        contracts_folder = "/home/cdsw/02_application/contracts"
        print(f"Creating contracts folder at: {contracts_folder}")
        os.makedirs(contracts_folder, exist_ok=True)
        print(f"Created {contracts_folder} directory. Please place your legal documents there and run again.")
        return None

    print(f"Using contracts folder: {os.path.abspath(contracts_folder)}")
    print(f"Current working directory: {os.getcwd()}")

    # Print directory contents for debugging
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(contracts_folder):
        level = root.replace(contracts_folder, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

    # Get all files in the contracts folder (including subfolders)
    contract_files = []

    print("\nScanning for legal document files (including subfolders)...")
    for root, dirs, files in os.walk(contracts_folder):
        print(f"Scanning directory: {root}")
        for filename in files:
            file_path = os.path.join(root, filename)
            # Check if file is of supported type
            ext = os.path.splitext(filename)[1].lower()
            print(f"Checking file: {filename} (extension: {ext})")
            if ext in ['.pdf', '.docx', '.txt']:
                print(f"Found supported legal file: {file_path}")
                contract_files.append(file_path)
            else:
                print(f"Skipping unsupported file: {file_path} (extension: {ext})")

    if not contract_files:
        print(f"No supported legal documents found in {contracts_folder} folder.")
        print("Supported formats: PDF, DOCX, TXT")
        # Try a direct file listing as a fallback
        print("\nAttempting direct file listing:")
        try:
            dir_contents = os.listdir(contracts_folder)
            print(f"Directory contents of {contracts_folder}:")
            for item in dir_contents:
                item_path = os.path.join(contracts_folder, item)
                if os.path.isfile(item_path):
                    print(f"  File: {item}")
                    # Check if it's a supported file but was missed somehow
                    ext = os.path.splitext(item)[1].lower()
                    if ext in ['.pdf', '.docx', '.txt']:
                        print(f"  Found missed supported file: {item_path}")
                        contract_files.append(item_path)
                elif os.path.isdir(item_path):
                    print(f"  Directory: {item}/")
        except Exception as e:
            print(f"Error listing directory: {e}")

        if not contract_files:
            return None

    print(f"Found {len(contract_files)} legal document(s) to process:")
    for i, file_path in enumerate(contract_files):
        print(f"  {i + 1}. {file_path}")

    # Process each document
    all_results = {}
    for i, file_path in enumerate(contract_files):
        doc_id = f"doc_{i + 1:03d}"
        filename = os.path.basename(file_path)

        print(f"\nProcessing legal document {i + 1}/{len(contract_files)}: {filename}")

        try:
            # Process the document
            print(f"Extracting and processing legal document...")
            processed_doc = document_processor.process_document(file_path, doc_id)

            # Store chunks in ChromaDB
            print(f"Storing legal document chunks in ChromaDB...")
            storage_result = chroma_storage.add_texts(
                processed_doc["chunks"],
                processed_doc["metadatas"],
                processed_doc["ids"]
            )

            print(f"Legal document processing result: {storage_result}")

            # Generate summary and analysis
            print("Generating legal document summary...")
            summary = document_processor.summarize(processed_doc["text"])

            print("Generating legal document analysis...")
            analysis = document_processor.analyze(processed_doc["text"])

            # Store summary in ChromaDB
            print("Storing legal document summary in ChromaDB...")
            summary_result = chroma_storage.add_summary(
                doc_id=doc_id,
                source=file_path,
                filename=filename,
                summary=summary,
                analysis=analysis
            )

            print(f"Legal summary storage result: {summary_result}")

            # Store results
            results = {
                "filename": filename,
                "processing": storage_result,
                "summary": summary,
                "analysis": analysis,
                "summary_storage": summary_result
            }

            all_results[doc_id] = results

            # Save results to file
            results_folder = "/home/cdsw/02_application/results"
            if not os.path.exists(results_folder):
                print(f"Creating results folder: {results_folder}")
                os.makedirs(results_folder)

            summary_file = os.path.join(results_folder, f"{os.path.splitext(filename)[0]}_summary.txt")
            print(f"Saving legal summary and analysis to: {summary_file}")

            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Legal Document: {filename}\n")
                f.write(f"Processed as: {doc_id}\n\n")
                f.write(f"SUMMARY:\n{results['summary']}\n\n")
                f.write(f"ANALYSIS:\n{results['analysis']}")

            print(f"Successfully saved legal summary and analysis to {summary_file}")

        except Exception as e:
            print(f"Error processing legal document {filename}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue

    # Print summary of processing
    print(f"\n==== Legal Document Processing Summary ====")
    print(f"Successfully processed {len(all_results)} out of {len(contract_files)} legal documents.")
    print(f"Results saved to the 'results' folder.")
    print(f"Document chunks and summaries stored in ChromaDB.")

    print("\n========== LEGAL DOCUMENT PROCESSING COMPLETED ==========\n")
    return all_results

    def main():
        """Main function."""
        print("Legal Document Processing System")
        print("---------------------------------------------")
        print("This system will process all legal documents in the 'contracts' folder (including subfolders).")
        print(f"Current working directory: {os.getcwd()}")

        # Verify YAML files exist at hardcoded locations
        print(f"Verifying YAML files...")
        if not os.path.exists(AGENTS_YAML_PATH):
            print(f"WARNING: Agents YAML file not found at {AGENTS_YAML_PATH}")
        else:
            print(f"Found agents YAML at {AGENTS_YAML_PATH}")

        if not os.path.exists(TASKS_YAML_PATH):
            print(f"WARNING: Tasks YAML file not found at {TASKS_YAML_PATH}")
        else:
            print(f"Found tasks YAML at {TASKS_YAML_PATH}")

        try:
            print("About to start legal document processing")
            process_documents()
            print("Legal document processing completed successfully.")
        except Exception as e:
            print(f"Error during legal document processing: {str(e)}")
            import traceback
            print(traceback.format_exc())
            print("Please check the error message and try again.")

    if __name__ == "__main__":
        print("Legal document processing script is starting...")
        main()
        print("Legal document processing script has finished.")