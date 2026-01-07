from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import shutil
import os
import PyPDF2
# from langchain.document_loaders import PyPDFDirectoryLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure maximum file upload size (50MB)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def home():
    return render_template('index.html')

# if os.path.exists("embaddings/"):
#     shutil.rmtree("embaddings/")

if not os.path.exists('uploads'):
    os.makedirs('uploads')
else:
    shutil.rmtree('uploads')
    os.makedirs('uploads')


# model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_name = "sentence-transformers/bert-base-nli-mean-tokens"
# model_name = "moka-ai/m3e-base"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf_embadings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print(hf_embadings)







def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    document = file_loader.load()
    return document



def chunc_data(docs,chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
        )
    docs=text_splitter.split_documents(docs)
    return docs

llm = OllamaLLM(model="llama3",  base_url="http://localhost:11434")










@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    embeddings_dir = None
    db = None
    try:
        # Create uploads directory if it doesn't exist
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']

        # Create a unique embeddings directory for each request
        import uuid
        import time
        timestamp = int(time.time())
        unique_id = uuid.uuid4().hex[:8]
        embeddings_dir = f"embeddings_{timestamp}_{unique_id}/"
        
        print(f"Using embeddings directory: {embeddings_dir}")

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and file.filename.endswith('.pdf'):
            filepath = os.path.join('uploads', file.filename)
            # dirpath = os.path.join('uploads')
            file.save(filepath)

            print(f"File saved to {filepath}")

            try:


                full_process_time_start = file_pre_pro_time_start = time.time()
                print(file_pre_pro_time_start)
                print(type(file_pre_pro_time_start))

                with open(filepath, 'rb') as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"


                documents = [Document(page_content=text, metadata={"source": "user_input"})]

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=50
                )

                chunked_documents = text_splitter.split_documents(documents)




                # doc=read_doc(dirpath)
                # # print(len(doc))
                # print("Doc reading done")

                # document = chunc_data(docs=doc)
                # # print(len(document))
                # print("Doc chunking done")


                # Create ChromaDB with unique directory (no persistence to avoid readonly issues)
                db = Chroma.from_documents(
                    chunked_documents, 
                    hf_embadings,
                    persist_directory=embeddings_dir
                )
                print(f"Database created in: {embeddings_dir}")

                retriever = db.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": 100,  # Increase the number of returned chunks
                    }
                )
                print("Doc embedding done")

                file_pre_pro_time_end = time.time()

                file_pre_pro_time = round(file_pre_pro_time_end - file_pre_pro_time_start, 2)
                print(f"File preprocessing time: {file_pre_pro_time} seconds")

                # author_prompt = ChatPromptTemplate.from_template("""You are an AI that extracts data from research paper text. From the given context, find all author names, keep them in the same order, and output them comma-separated. Don’t include affiliations, titles, or extra text — only the names.
                #                                                         \n Context : {context}
                #                                                         \n Input : {input}
                #                                         """)
                # Create prompt template



                author_prompt = ChatPromptTemplate.from_template("""
                    You are an AI assistant designed to analyze research papers and extract meaningful, relevant information. Your purpose is to help users by reading the given paper or its context and answering specific questions based on its content I am providing you with the context of a research paper. Using only this provided content, carefully find and extract accurate information to answer the following questions.  Do not make assumptions or add external knowledge. If any answer is not clearly mentioned in the paper, reply with: "Not mentioned in the provided context." Keep answers concise, factual, and directly quoted or paraphrased from the paper.:
                    Context: {context}
                    Question: {input}
                """)
                



                
                print("Prompt done")
                # print(context)

                author_document_chain = create_stuff_documents_chain(llm,author_prompt)
                print("Chain done")


                # # Add this before creating the retrieval chain
                # print("Retrieved Documents:")
                # retrieved_docs = retriever.get_relevant_documents("From the given context, find all author names, keep them in the same order")
                # for i, doc in enumerate(retrieved_docs):
                #     print(f"\nDocument {i+1}:")
                #     print("Content:", doc.page_content[:200], "...")  # Print first 200 chars
                #     print("Metadata:", doc.metadata)


                retrieval_chain=create_retrieval_chain(retriever,author_document_chain)
                print("Final Chain done")



                abstract_time_start = time.time()
                abstract_response = retrieval_chain.invoke({
                    "input": "Read the given research paper carefully and extract or generate the abstract from it. If the paper already contains an abstract section, provide only that text exactly as it appears. If no explicit abstract is found, summarize the paper briefly to create a meaningful abstract based on its content. Do not include any other sections, headings, or extra text — provide only the abstract."
                })
                print("Abstract response done")
                abstract_time_end = time.time()
                abstract_time = round(abstract_time_end - abstract_time_start, 2)
                print(f"Abstract extraction time: {abstract_time} seconds")



                summary_time_start = time.time()
                summary_response = retrieval_chain.invoke({
                    "input": "Read the entire research paper carefully and provide a concise summary of the whole paper in simple and clear language. The summary should capture the main objectives, methods, key findings, and conclusions of the paper. Avoid technical jargon and long explanations — keep it short, easy to understand, and focused only on the core ideas of the paper. Do not include any references, author names, or section headings — provide only the summary."
                })
                print("Summary response done")
                summary_time_end = time.time()
                summary_time = round(summary_time_end - summary_time_start, 2)
                print(f"Summary extraction time: {summary_time} seconds")



                tech_time_start = time.time()
                tech_response = retrieval_chain.invoke({
                    "input": "Read the entire research paper carefully and extract all details related to the technologies, tools, programming languages, frameworks, algorithms, or platforms used in the work. Clearly identify what technology stack the paper is based on or what technologies it applies or proposes. Summarize this information in simple and clear language without adding any unrelated content. Do not include author names, references, or other sections — provide only the information about the technologies used or discussed in the paper."
                })
                print("Tech response done")
                tech_time_end = time.time()
                tech_time = round(tech_time_end - tech_time_start, 2)
                print(f"Technology extraction time: {tech_time} seconds")



                challenges_time_start = time.time()
                challenges_response = retrieval_chain.invoke({
                    "input": "Read the entire research paper carefully and identify all the challenges, limitations, or difficulties mentioned in the study or project. If the paper does not explicitly list any challenges, analyze the content and infer the possible challenges or issues that the researchers or project might have faced based on the methods, technologies, and objectives described. Provide the challenges in clear, simple, and concise language. Do not include author names, references, or other sections — provide only the list or summary of challenges faced or potentially faced in the research."
                })
                print("Challenges response done")
                challenges_time_end = time.time()
                challenges_time = round(challenges_time_end - challenges_time_start, 2)
                print(f"Challenges extraction time: {challenges_time} seconds")


                benefits_time_start = time.time()
                benefits_response = retrieval_chain.invoke({
                    "input": "Read the entire research paper carefully and identify all the benefits, advantages, or positive outcomes mentioned in the study or project. Clearly explain who can benefit from this research (for example, individuals, organizations, industries, or society) and how they benefit (in what way or form). If the paper does not explicitly mention the benefits, analyze the content and infer the possible benefits based on the objectives, methods, and results. Present the answer in clear, simple, and concise language. Do not include author names, references, or other sections — provide only the benefits and who gains them, along with how and in what form."
                })
                print("Benefits response done")
                benefits_time_end = time.time()
                benefits_time = round(benefits_time_end - benefits_time_start, 2)
                print(f"Benefits extraction time: {benefits_time} seconds")


                conclusion_time_start = time.time()
                conclusion_response = retrieval_chain.invoke({
                    "input": "Read the entire research paper carefully and extract the conclusion section exactly as it appears. If the paper does not explicitly include a conclusion, analyze the content and create a concise conclusion based on the overall objectives, methods, findings, and results of the paper. The conclusion should be short, clear, and written in simple language, summarizing the key takeaway of the research. Do not include author names, references, or other sections — provide only the conclusion."
                })
                print("Conclusion response done")
                conclusion_time_end = time.time()
                conclusion_time = round(conclusion_time_end - conclusion_time_start, 2)
                print(f"Conclusion extraction time: {conclusion_time} seconds")


                meta_time_start = time.time()
                meta_response = retrieval_chain.invoke({
                    "input": "Read the entire research paper carefully and extract any available metadata information. This may include the paper title, author names, publication year, journal or conference name, DOI, keywords, affiliations, and any other relevant metadata found. If some metadata fields are missing, provide only the ones available in the paper — do not generate or assume any details. Present the extracted metadata in a clear and organized format. Do not include the paper's main content or any extra text — provide only the metadata."
                })
                print("Meta response done")
                meta_time_end = time.time()
                meta_time = round(meta_time_end - meta_time_start, 2)
                print(f"Meta extraction time: {meta_time} seconds")


                # Clean up database and temporary directory
                try:
                    # Properly close ChromaDB
                    if db:
                        try:
                            # Try to reset client if available
                            if hasattr(db, '_client') and db._client:
                                db._client.reset()
                        except Exception as db_error:
                            print(f"Warning: Could not reset DB client: {db_error}")
                        
                        # Delete database reference
                        db = None
                        
                    # Clear retriever reference
                    retriever = None
                    
                    # Force garbage collection multiple times
                    import gc
                    for _ in range(3):
                        gc.collect()
                    
                    # Wait longer for file handles to be released
                    import time
                    time.sleep(1.0)
                    
                    # Clean up the temporary embeddings directory
                    if embeddings_dir and os.path.exists(embeddings_dir):
                        try:
                            # Force close any remaining file handles on Windows/Linux
                            import platform
                            if platform.system() == "Windows":
                                import subprocess
                                subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                                             capture_output=True, check=False)
                            
                            # Multiple attempts to remove directory
                            max_attempts = 5
                            for attempt in range(max_attempts):
                                try:
                                    shutil.rmtree(embeddings_dir)
                                    print(f"Cleaned up embeddings directory: {embeddings_dir}")
                                    break
                                except OSError as e:
                                    if attempt < max_attempts - 1:
                                        print(f"Attempt {attempt + 1} failed, retrying...")
                                        time.sleep(0.5)
                                        gc.collect()
                                    else:
                                        print(f"Could not remove embeddings directory after {max_attempts} attempts: {e}")
                                        # Try to rename instead of delete
                                        try:
                                            import uuid
                                            backup_name = f"{embeddings_dir}_backup_{uuid.uuid4().hex[:8]}"
                                            os.rename(embeddings_dir, backup_name)
                                            print(f"Renamed embeddings directory to: {backup_name}")
                                        except Exception as rename_error:
                                            print(f"Could not rename directory: {rename_error}")
                        except Exception as dir_error:
                            print(f"Error during directory cleanup: {dir_error}")
                        
                except Exception as cleanup_error:
                    print(f"Warning during cleanup: {cleanup_error}")
                
                # print("Full Chain Input:", response)  # This will show the complete context being used
                print("All Response done")

                with open(filepath, 'rb') as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"

                # Clean up the file after processing
                os.remove(filepath)

                full_process_time_end = time.time()
                full_process_time = round(full_process_time_end - full_process_time_start, 2)
                print(f"Full processing time: {full_process_time} seconds")

                # Function to clean markdown formatting from text
                def clean_markdown_text(text):
                    import re
                    if not text:
                        return text
                    
                    # Remove bold markdown (**text**)
                    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
                    
                    # Remove italic markdown (*text* or _text_)
                    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
                    text = re.sub(r'_(.+?)_', r'\1', text)
                    
                    # Remove headers (# text)
                    text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)
                    
                    # Remove inline code (`text`)
                    text = re.sub(r'`(.+?)`', r'\1', text)
                    
                    # Remove strikethrough (~~text~~)
                    text = re.sub(r'~~(.+?)~~', r'\1', text)
                    
                    # Clean up excessive whitespace
                    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
                    
                    # Clean up star characters( * text )
                    text = re.sub(r'^\*\s+(.+)$', r'\1', text, flags=re.MULTILINE)
                    
                    return text.strip()

                return jsonify({
                    "success": True,
                    "filename": file.filename,
                    # "text": text,
                    "abstract": clean_markdown_text(abstract_response['answer']),
                    "summary": clean_markdown_text(summary_response['answer']),
                    "tech": clean_markdown_text(tech_response['answer']),
                    "challenges": clean_markdown_text(challenges_response['answer']),
                    "benefits": clean_markdown_text(benefits_response['answer']),
                    "conclusion": clean_markdown_text(conclusion_response['answer']),
                    "meta": clean_markdown_text(meta_response['answer']),
                    "full_process_time_seconds": full_process_time,
                    "file_preprocessing_time_seconds": file_pre_pro_time,
                    "abstract_extraction_time_seconds": abstract_time,
                    "summary_extraction_time_seconds": summary_time,
                    "tech_extraction_time_seconds": tech_time,
                    "challenges_extraction_time_seconds": challenges_time,
                    "benefits_extraction_time_seconds": benefits_time,
                    "conclusion_extraction_time_seconds": conclusion_time,
                    "meta_extraction_time_seconds": meta_time
                    # "Input": response,
                })
                
            except Exception as e:
                # Clean up on error with robust cleanup
                try:
                    # Clean up database
                    if 'db' in locals() and db:
                        try:
                            if hasattr(db, '_client') and db._client:
                                db._client.reset()
                        except:
                            pass
                        db = None
                    
                    # Clean up embeddings directory
                    if 'embeddings_dir' in locals() and embeddings_dir and os.path.exists(embeddings_dir):
                        import time
                        import gc
                        gc.collect()
                        time.sleep(0.5)
                        
                        try:
                            shutil.rmtree(embeddings_dir)
                            print(f"Cleaned up embeddings directory after error: {embeddings_dir}")
                        except Exception as cleanup_err:
                            print(f"Could not clean embeddings after error: {cleanup_err}")
                            try:
                                # Try renaming if deletion fails
                                import uuid
                                backup_name = f"{embeddings_dir}_error_{uuid.uuid4().hex[:8]}"
                                os.rename(embeddings_dir, backup_name)
                                print(f"Renamed embeddings directory to: {backup_name}")
                            except:
                                pass
                except Exception as outer_cleanup_error:
                    print(f"Error during error cleanup: {outer_cleanup_error}")
                
                return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500

        return jsonify({"error": "Invalid file type. Please upload a PDF file."}), 400

    except Exception as e:
        # Clean up on outer error with robust cleanup
        try:
            if 'embeddings_dir' in locals() and embeddings_dir and os.path.exists(embeddings_dir):
                import time
                import gc
                gc.collect()
                time.sleep(0.5)
                
                try:
                    shutil.rmtree(embeddings_dir)
                    print(f"Cleaned up embeddings directory after outer error: {embeddings_dir}")
                except Exception as cleanup_err:
                    print(f"Could not clean embeddings after outer error: {cleanup_err}")
                    try:
                        # Try renaming if deletion fails
                        import uuid
                        backup_name = f"{embeddings_dir}_outer_error_{uuid.uuid4().hex[:8]}"
                        os.rename(embeddings_dir, backup_name)
                        print(f"Renamed embeddings directory to: {backup_name}")
                    except:
                        pass
        except Exception as cleanup_error:
            print(f"Error during outer error cleanup: {cleanup_error}")
        
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    
    