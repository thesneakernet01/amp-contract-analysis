# Legal Document Analysis System

A comprehensive system for analyzing, comparing, and managing legal documents using agentic AI tools. Built for Cloudera Applied Machine Learning Prototypes (AMP), this application provides advanced document processing capabilities for legal professionals.

## Overview

This Legal Document Analysis System uses AI to process, analyze, and compare legal documents. It provides detailed summaries, extracts key legal provisions, identifies legal definitions, assesses risks, and can compare multiple documents to identify similarities and differences.

## Features

- **Document Processing**: Upload and process legal documents in PDF, DOCX, and TXT formats
- **Legal Summaries**: Generate concise summaries highlighting key legal information
- **Legal Analysis**: Identify and extract key provisions, obligations, and rights
- **Legal Definitions**: Extract defined terms from legal documents with explanations
- **Risk Assessment**: Analyze legal documents for potential risks with severity ratings
- **Document Comparison**: Compare multiple legal documents to identify similarities and differences
- **Legal Compliance**: Check documents for compliance with relevant regulations
- **Vector Database**: Store and search legal documents using semantic similarity
- **Advanced Search**: Search for specific legal terms or concepts across all documents

## Hardware Requirements

- **Minimum Requirements**: 4 CPU cores, 8GB RAM
- **Recommended**: 8+ CPU cores, 16GB+ RAM for larger document collections

Note: Hardware resource allocation is automatically handled by the `.project-metadata.yaml` configuration file.

## Installation in Cloudera AI

1. Create a new project in Cloudera AI by providing the HTTP Git URL for this repository
2. The hardware resources (minimum 4 CPU, 8GB RAM) will be automatically allocated as specified in the `.project-metadata.yaml` file
3. Enter your OpenAI API key in the AMP deployment wizard when prompted
   - The key will be automatically set as the `OPENAI_API_KEY` environment variable
   - You can also update the API key later through the Settings tab in the application

Once deployed, the application will be automatically started and available through the Cloudera AI interface. No manual installation or startup steps are required.

## Usage

### Processing Documents

1. Place legal documents in the `contracts` folder
2. Navigate to the application in your browser
3. Click "Process Legal Documents in 'contracts' Folder" to batch process, or use the file uploader to process individual files
4. View the generated summaries and analysis

### Analyzing Documents

The system provides several types of analysis:
- **Summary & Analysis**: Overall summary and key legal analysis
- **Legal Provisions**: Extraction of key provisions, obligations, and rights
- **Legal Definitions**: Glossary of defined terms in the document
- **Risk Assessment**: Risk analysis with severity and likelihood ratings
- **Similar Documents**: Find and compare similar documents in the database

### Searching Documents

- Use the search functionality in the sidebar to find specific legal terms or concepts
- Adjust similarity threshold and maximum results to refine searches
- Choose between searching in document chunks or summaries

## System Architecture

This system uses a combination of:
- **CrewAI**: Orchestrates specialized AI agents for different legal tasks
- **ChromaDB**: Vector database for storing and retrieving document chunks and summaries
- **Langchain**: For text splitting and document processing
- **Streamlit**: Web interface for interaction with the system

## Specialized Legal Agents

The system employs multiple specialized agents:
- **Document Processor**: Handles initial document processing
- **Document Summarizer**: Creates concise legal summaries
- **Legal Analyzer**: Identifies key provisions, obligations, and rights
- **Document Comparer**: Analyzes similarities and differences between documents
- **Contract Reviewer**: Identifies risks in contracts
- **Legal Compliance Checker**: Assesses regulatory compliance
- **Legal Terminology Extractor**: Identifies and explains legal terms
- **Legal Risk Assessor**: Evaluates legal risks with severity ratings

## File Structure

- `app.py`: Main Streamlit application
- `main.py`: Core document processing logic
- `agents.yaml`: Configuration for CrewAI agents
- `tasks.yaml`: Configuration for CrewAI tasks
- `contracts/`: Default folder for legal documents
- `results/`: Generated summaries and analysis
- `chromadb/`: Vector database storage
- `config/`: Configuration files

## License

Apache 2.0 License - See LICENSE file for details

---

© Cloudera, Inc. 2021