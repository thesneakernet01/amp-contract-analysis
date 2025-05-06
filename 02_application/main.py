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