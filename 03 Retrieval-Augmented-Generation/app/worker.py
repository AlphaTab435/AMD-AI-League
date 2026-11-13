import sys
import os
import io
from unittest import result
from PIL import Image
from pathlib import Path
import pymupdf
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import chromadb
from hashlib import sha256
from config import HOST, PORT
import socket
import json
import textwrap

# print(dir(os)) # To list methods in a library
# help(os.path) # Too get readable docs about a library or its method
# # python -m pydoc <module_name>
# python -c "import sys, hashlib; print('Standard library modules are working correctly!')"
# rocm-smi
# rocm-smi --showmeminfo vram

# importing neceser paths
current_dir = Path(__file__).resolve().parent
db_dir = current_dir / "db"
target_dir_vlm = current_dir.parent / "models" / "qwen2.5-vl-3b"
target_dir_embedding = current_dir.parent / "models" / "qwen3-embedding-0.6b"
target_dir_mc3_corpus = current_dir.parent / "mc3-starter-kit" / "mc3-corpus"
# target_dir_test = current_dir.parent / "test"
# candy_image_path = target_dir_test / "candy.JPG"
print(f"target_dir_vlm: {target_dir_vlm}")
print(f"target_dir_embedding: {target_dir_embedding}")
print(f"target_dir_mc3_corpus: {target_dir_mc3_corpus}")
print(f"db_dir: {db_dir}")
# print(f"candy_image_path: {candy_image_path}")
# print(f"candy_image_path.exists(): {candy_image_path.exists()}")
print("=" * 10)

# load models
vlm_pipe = pipeline("image-text-to-text", model=str(target_dir_vlm))
embed_model = SentenceTransformer(str(target_dir_embedding))

# Initialize vector database once
db_dir.mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=str(db_dir))
# Maps canonical corpus paths to their Chroma collection objects
indexed_collections = {}

def process_index(path):
    all_files_data = []
    if os.path.isfile(path):
        print(f"'{path}' is a file.")
        data = process_file(path)
        if data:
            all_files_data.append(data)
    elif os.path.isdir(path):
        print(f"'{path}' is a folder. Walking through contents:")
        for root, dirs, files in os.walk(path):
            print(f"root:{root}\ndirs:{dirs}\nfiles:{files}")
            for file in files:
                full_file_path = os.path.join(root, file) 
                print(f"os.path.join(root, file):{full_file_path}")
                data = process_file(full_file_path)
                if data:
                    all_files_data.append(data)
    else:
        print(f"'{path}' does not exist or is a special type.")
    all_chunked_data = chunker(all_files_data)
    embeddings = generate_embeddings(all_chunked_data)
    collection = create_vector_db( path, all_chunked_data, embeddings )

    return collection


def process_file(file_path):
    # Mapping extensions to a tuple of (File Type Label, Parser Function)
    parser_mapping = {
        '.pdf':  ('PDF', pdf_parser),
        '.docx': ('Word', docx_parser),
        '.xlsx': ('Excel', xlsx_parser),
        '.csv':  ('CSV', csv_parser),
        '.txt':  ('Text/Log', text_parser),
        '.log':  ('Text/Log', text_parser),
        '.py':   ('Code', code_parser),
        '.png':  ('Image', vision_model_parser),
        '.jpg':  ('Image', vision_model_parser),
        '.jpeg': ('Image', vision_model_parser),
    }
    
    # Safely extract the extension in lowercase
    _, ext = os.path.splitext(file_path.lower())
    
    try:
        if ext in parser_mapping:
            label, parser_func = parser_mapping[ext]
            print(f"Parsing {label}: {file_path}")
            content = parser_func(file_path) # Call the actual parser function here
            return {
                "source": str(file_path),
                "extension": ext,
                "content": content
            }
        else:
            print(f"UNKNOWN file type: {file_path}")
    except Exception as error:
        print(f"Skipping unreadable file {file_path}: {error}")

def pdf_data_extractor(file_path):
    all_text_with_placeholders = []
    images_by_page={}
        
    with pymupdf.open(file_path) as doc:
        for page_num, page in enumerate(doc):
            page_items=[]
            images_by_page[page_num] = []
            # extract tex blocks
            raw_blocks = page.get_text("blocks")
            # print(f"raw_blocks: {raw_blocks}")

            for b in raw_blocks:
                text_content = b[4].strip()
                if text_content:  # Only grab actual text strings
                    # Structure: (y0 coordinate, type flag, actual content)
                    page_items.append((b[1], "text", text_content))

            image_infos = page.get_image_info(xrefs=True)
            # print(f"image_infos: {image_infos}")
            # page_images = page.get_images()
            # print(f"page_images: {page_images}")
            for img in image_infos:
                xref = img.get("xref", 0)
                if xref == 0:
                    continue
                rect = img["bbox"]  # (x0, y0, x1, y1)
                placeholder = f"[IMAGE_PLACEHOLDER_XREF_{xref}]"
                # Append using the image's top vertical coordinate (rect[1] is y0)
                page_items.append((rect[1], "image", placeholder))

                # Extract raw bytes for your VLM pipeline
                base_image = doc.extract_image(xref)
                images_by_page[page_num].append({
                    "placeholder": placeholder,
                    "bytes": base_image["image"],
                    "ext": base_image["ext"]
                })
                # filename = f"image_{xref}.{base_image['ext']}"
                # with open(filename, "wb") as f:
                #     f.write(base_image["image"])

            page_items.sort(key=lambda x: x[0])

            page_content = []
            for y0, item_type, content in page_items:
                if item_type == "text":
                    page_content.append(content)
                elif item_type == "image":
                    page_content.append(f"\n{content}\n")
            all_text_with_placeholders.append("\n".join(page_content))
            
    final_output = "".join(
        f"--- PAGE {page_num + 1} ---\n{page_text}\n"
        for page_num, page_text in enumerate(all_text_with_placeholders)
    )
    return final_output, images_by_page

def pdf_parser(file_path):
    text_output, images_by_page = pdf_data_extractor(file_path)
    
    for page_num, images in images_by_page.items():
        for img_info in images:
            placeholder = img_info["placeholder"]
            try:
                # Convert in-memory bytes to PIL Image without writing to disk
                image = Image.open(io.BytesIO(img_info["bytes"])).convert("RGB")
                extracted_text = run_vlm_on_image(image)
                vlm_content = f"[IMAGE EXTRACTED DATA: {extracted_text}]"
            except Exception as error:
                print(f"Error processing PDF image {placeholder} on page {page_num + 1}: {error}")
                vlm_content = f"[IMAGE EXTRACTED DATA: Processing failed - {error}]"
            
            text_output = text_output.replace(placeholder, vlm_content)
            
    return text_output       

def docx_data_extractor(file_path):
    doc = Document(file_path)
    content_parts = []
    images_data = []
    image_counter = 1

    def parse_node_for_media_and_text(node):
        nonlocal image_counter
        parts = []
        
        # Traverse all descendants of the node using standard iteration
        for child in node.iter():
            tag = child.tag.split('}')[-1] if isinstance(child.tag, str) else ""
            
            # 1. Extract text nodes
            if tag == 't':
                if child.text:
                    parts.append(child.text)
                    
            # 2. Extract images/drawings via blip elements
            elif tag == 'blip':
                r_embed = None
                for attr_name, attr_val in child.attrib.items():
                    if attr_name.endswith('embed'):
                        r_embed = attr_val
                        break
                
                if r_embed and r_embed in doc.part.related_parts:
                    image_part = doc.part.related_parts[r_embed]
                    image_bytes = image_part.blob
                    ext = image_part.content_type.split('/')[-1]
                    if ext == 'jpeg':
                        ext = 'jpg'
                    
                    placeholder = f"[IMAGE_PLACEHOLDER_DOCX_{image_counter}]"
                    parts.append(f"\n{placeholder}\n")
                    
                    images_data.append({
                        "placeholder": placeholder,
                        "r_id": r_embed,
                        "bytes": image_bytes,
                        "ext": ext
                    })
                    image_counter += 1
                    
        return "".join(parts)

    def process_paragraph(p):
        p_text_parts = []
        
        # Iterate over direct XML child elements of the paragraph in strict order
        for child in p._p:
            tag = child.tag.split('}')[-1] if isinstance(child.tag, str) else ""
            
            if tag == 'r':  # Standard text run
                p_text_parts.append(parse_node_for_media_and_text(child))
                
            elif tag == 'hyperlink':  # Embedded hyperlink
                link_text_parts = []
                for sub_child in child:
                    sub_tag = sub_child.tag.split('}')[-1] if isinstance(sub_child.tag, str) else ""
                    if sub_tag == 'r':
                        link_text_parts.append(parse_node_for_media_and_text(sub_child))
                display_text = "".join(link_text_parts)
                
                # Resolve target URL from relationship ID
                r_id = None
                for attr_name, attr_val in child.attrib.items():
                    if attr_name.endswith('id'):
                        r_id = attr_val
                        break
                
                url = ""
                if r_id and r_id in p.part.rels:
                    url = p.part.rels[r_id].target_ref
                
                if url:
                    p_text_parts.append(f"[{display_text}]({url})")
                else:
                    p_text_parts.append(display_text)
                    
            elif tag == 'drawing':  # Block-level drawing/image outside runs
                p_text_parts.append(parse_node_for_media_and_text(child))
                        
        return "".join(p_text_parts)

    # Iterate through body elements (paragraphs, tables) in exact document sequence
    for element in doc.element.body:
        tag = element.tag.split('}')[-1] if isinstance(element.tag, str) else ""
        if tag == 'p':
            p = Paragraph(element, doc)
            text = process_paragraph(p)
            if text.strip():
                content_parts.append(text)
                
        elif tag == 'tbl':
            table = Table(element, doc)
            table_text = []
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells]
                table_text.append(" | ".join(row_cells))
            if table_text:
                content_parts.append("\n".join(table_text))

    final_output = "\n".join(content_parts)
    return final_output, images_data

def docx_parser(file_path):
    text_output, images_data = docx_data_extractor(file_path)
    
    for img_info in images_data:
        placeholder = img_info["placeholder"]
        try:
            # Convert in-memory bytes to PIL Image without writing to disk
            image = Image.open(io.BytesIO(img_info["bytes"])).convert("RGB")
            extracted_text = run_vlm_on_image(image)
            vlm_content = f"[IMAGE EXTRACTED DATA: {extracted_text}]"
        except Exception as error:
            print(f"Error processing DOCX image {placeholder}: {error}")
            vlm_content = f"[IMAGE EXTRACTED DATA: Processing failed - {error}]"
        
        text_output = text_output.replace(placeholder, vlm_content)
        
    return text_output

def xlsx_parser(file_path):
    """
    Production Excel parser for RAG pipelines.
    Preserves raw table structure as Markdown blocks per sheet,
    preventing header-corruption on arbitrary layouts.
    """
    # Load raw data without assuming row 0 is a header (header=None for pandas read)
    sheets_data = pd.read_excel(file_path, sheet_name=None, header=None)
    content_parts = []

    for sheet_name, df in sheets_data.items():
        # Drop completely empty rows and columns
        df = df.dropna(how='all').dropna(how='all', axis=1).fillna("")
        if df.empty:
            continue

        content_parts.append(f"### SHEET: {sheet_name}\n")
        
        # FIX: tabulate uses `headers` (plural)
        markdown_table = df.to_markdown(index=False, headers=[])
        content_parts.append(markdown_table)
        content_parts.append("\n")

    final_output = "\n".join(content_parts)
    return final_output

def csv_parser(file_path):
    """
    Production CSV parser for RAG pipelines.
    Renders raw tabular layout into Markdown without assuming fixed headers.
    """
    file_name = os.path.basename(file_path)
    df = pd.read_csv(file_path, header=None)
    
    # Drop completely empty rows and columns
    df = df.dropna(how='all').dropna(how='all', axis=1).fillna("")
    if df.empty:
        return f"--- CSV FILE: {file_name} (EMPTY) ---"

    content_parts = [
        f"### CSV FILE: {file_name}\n",
        # FIX: tabulate uses `headers` (plural)
        df.to_markdown(index=False, headers=[])
    ]

    return "\n".join(content_parts)

def text_parser(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        text=file.read()
    # print(text)
    return text

def code_parser(file_path):
    """Parses code files and wraps them in Markdown code blocks for better LLM context."""
    file_name = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        code_content = f.read()
    return f"### CODE FILE: {file_name}\n```python\n{code_content}\n```"

def run_vlm_on_image(image):
    """Core helper to run VLM pipeline on a PIL Image object."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text", 
                    "text": "Extract all readable text, describe any charts, tables, or diagrams, and provide a comprehensive summary of this image for a search database."
                }
            ]
        }
    ]
    result = vlm_pipe(text=messages)
    return result[0]["generated_text"][-1]["content"]

def vision_model_parser(file_path):
    """Parses standalone image files from disk."""
    with Image.open(file_path) as img:
        image = img.convert("RGB")
        return run_vlm_on_image(image)

def chunker(all_files_data, chunk_size=1000):
    
    chunked_data = []
    
    for file_data in all_files_data:
        content = file_data["content"]
        ext = file_data["extension"]
        
        if not content or not content.strip():
            continue
        if ext in [".xlsx", ".csv", ".py"]:
            chunks = [content]
        else:
            paragraphs = content.splitlines()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for paragraph in paragraphs:
                if not paragraph.strip():
                    continue
                if current_length + len(paragraph) > chunk_size and current_chunk:
                    chunks.append("\n".join(current_chunk))
                    # Carry previous chunk's ending into the next chunk
                    current_chunk = [current_chunk[-1]]
                    current_length = len(current_chunk[0])
                current_chunk.append(paragraph)
                current_length += len(paragraph)

            if current_chunk:
                chunks.append("\n".join(current_chunk))

        for index, chunk in enumerate(chunks):
            chunked_data.append({
                "source": file_data["source"],
                "extension": ext,
                "chunk_index": index,
                "content": chunk
            })

    return chunked_data

def generate_embeddings(all_chunked_data, batch_size=8):

    all_texts = [item["content"] for item in all_chunked_data]

    embeddings = embed_model.encode_document(
        all_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    
    print(f"Chunks: {len(all_texts)}")
    print(f"Embedding shape: {embeddings.shape}")

    return embeddings

def create_vector_db(index_path, all_chunked_data, embeddings):

    collection_name = get_collection_name(index_path)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "index_path": str(Path(index_path).resolve())
        }
    )

    ids = [f"chunk_{i}" for i in range(len(all_chunked_data))]
    documents = [
        item["content"] for item in all_chunked_data
    ]
    metadatas = [
        {
            "source": item["source"],
            "extension": item["extension"],
            "chunk_index": item["chunk_index"]
        }
        for item in all_chunked_data
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )

    # Save the collection handle against its index path
    indexed_collections[str(Path(index_path).resolve())] = collection

    print(f"Collection: {collection_name}")
    print(f"Vectors stored: {collection.count()}")
    print(f"Database location: {db_dir}")

    return collection

def get_collection_name(index_path):
    path = str(Path(index_path).resolve())
    path_hash = sha256(path.encode("utf-8")).hexdigest()[:16]

    return f"corpus_{path_hash}"

def find_covering_collection(path):
    requested_path = Path(path).resolve()
    for indexed_path, collection in indexed_collections.items():
        existing_path = Path(indexed_path)
        if requested_path.is_relative_to(existing_path):
            return collection
    return None

# process_index(target_dir_test)
# process_index(target_dir_test / "encrypted-files")
# process_index(target_dir_mc3_corpus)

def validate_citations(query_text, answer, citations, documents, metadatas):

    if not answer or not isinstance(citations, list):
        return []

    # Map each retrieved source path to its document content.
    source_documents = {}

    for index, metadata in enumerate(metadatas):
        source = metadata.get("source", "")
        source_documents[source] = documents[index]

    # Keep only valid, unique citation paths.
    validated_citations = []

    for citation in citations:
        if citation in source_documents and citation not in validated_citations:
            validated_citations.append(citation)
        else:
            print(f"Invalid or duplicate citation: {citation}")

    # Test whether each citation is necessary.
    for citation in validated_citations.copy():

        # Get the other currently retained citations.
        other_citations = [
            source for source in validated_citations
            if source != citation
        ]

        # A citation cannot be removed if no other evidence remains.
        if not other_citations:
            continue

        # Build context without the document being tested.
        remaining_context = "\n\n---\n\n".join(
            f"SOURCE: {source}\n"
            f"CONTENT:\n{source_documents[source]}"
            for source in other_citations
        )

        # validation_prompt = textwrap.dedent(f"""
        #     You are validating whether a document citation is necessary.

        #     ORIGINAL QUESTION:
        #     {query_text}

        #     PROPOSED ANSWER:
        #     {answer}

        #     DOCUMENTS REMAINING AFTER EXCLUDING ONE CITATION:
        #     {remaining_context}

        #     Determine whether the remaining documents ALONE provide
        #     sufficient evidence to establish the proposed answer
        #     to the original question.

        #     Rules:

        #     1. Use ONLY the remaining documents. Do not use outside
        #        knowledge or assume missing information.

        #     2. Check every condition and requirement in the original
        #        question, not just whether the proposed answer appears
        #        somewhere in the documents.

        #     3. Consider relationships between documents. A document
        #        may provide necessary intermediate evidence even when
        #        it does not contain the final answer.

        #     4. If the remaining documents independently establish
        #        the complete proposed answer, return true.

        #     5. If any required information or relationship is missing,
        #        return false.

        #     Return ONLY valid JSON:

        #     {{
        #         "answer_supported": true
        #     }}

        #     Use false instead of true when the evidence is insufficient.
        # """).strip()

        # ========================

#         validation_prompt = textwrap.dedent(f"""
#     You are evaluating whether one document citation is necessary.

#     ORIGINAL QUESTION:
#     {query_text}

#     PROPOSED ANSWER:
#     {answer}

#     REMAINING DOCUMENTS:
#     {remaining_context}

#     TASK:
#     Determine whether the remaining documents are sufficient
#     to establish the proposed answer to the original question
#     WITHOUT the excluded document.

#     IMPORTANT DISTINCTIONS:

#     1. A document is not necessary merely because it discusses
#        the same topic, mentions the same identifier, or provides
#        background information.

#     2. A document is necessary only when it supplies information
#        required to establish the answer or connect required
#        evidence across documents.

#     3. Evaluate the complete question, including all its
#        conditions and relationships.

#     4. Multiple documents may form an evidence chain.
#        Preserve every document required to complete that chain.

#     5. Do not require supporting documents to repeat or confirm
#        information already established by another document.

#     DECISION:

#     - Return true if the remaining documents independently
#       establish the complete proposed answer.
#     - Return false if any required information or relationship
#       is missing.

#     Return ONLY valid JSON:

#     {{
#         "answer_supported": true
#     }}

#     Replace true with false when the evidence is insufficient.
# """).strip()
    
    # ================

        validation_prompt = textwrap.dedent(f"""
        You are validating whether a document citation is necessary.
    
        ORIGINAL QUESTION:
        {query_text}
    
        PROPOSED ANSWER:
        {answer}
    
        REMAINING DOCUMENTS:
        {remaining_context}
    
        TASK:
    
        Determine whether the remaining documents, without the
        excluded citation, contain enough evidence to establish
        the complete proposed answer to the original question.
    
        Follow these steps:
    
        1. Identify the exact information requested by the question,
           including all conditions, identifiers, and relationships.
    
        2. Find the evidence for each required piece of information
           in the remaining documents.
    
        3. If the question requires connecting information across
           documents, identify the complete evidence chain.
    
        4. Check whether every link in that evidence chain is
           established by the remaining documents.
    
        5. Do not require unnecessary background documents,
           explanations, or repeated confirmation.
    
        6. Do not assume missing relationships or use outside
           knowledge.
    
        DECISION:
    
        Return true ONLY if the remaining documents establish
        every required piece of information and relationship
        needed to answer the original question completely.
    
        Otherwise, return false.
    
        Return ONLY valid JSON:
    
        {{
            "answer_supported": true
        }}
    
        Replace true with false when the evidence is insufficient.
        """).strip()

        validation_messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": validation_prompt
                }
            ]
        }]

        validation_result = vlm_pipe(text=validation_messages)

        validation_output = (
            validation_result[0]["generated_text"][-1]["content"].strip()
        )

        if validation_output.startswith("```"):
            validation_output = (
                validation_output
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )

        print(f"Validating citation: {citation}")
        print(f"Validation output: {validation_output}")

        try:
            validation_response = json.loads(validation_output)

            answer_supported = validation_response.get(
                "answer_supported"
            )

        except (json.JSONDecodeError, AttributeError):
            print("Could not parse validation response. Keeping citation.")
            continue

        if answer_supported is True:
            print(f"Removing unnecessary citation: {citation}")
            validated_citations.remove(citation)

        elif answer_supported is False:
            print(f"Keeping necessary citation: {citation}")

        else:
            print("Invalid validation result. Keeping citation.")

    print(f"Validated citations: {validated_citations}")

    return validated_citations

def resolve_query(collection, query_text, n_results=5):
    print(f"===Starting Resolving query Function===")
    # print(f"Querying collection: {collection}")
    # print(f"Querying collection.name: {collection.name}")
    print(f"Query text: {query_text}")

    query_embedding = embed_model.encode_query(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    result = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    print(f"=result start=")
    print(result)
    print(f"=result end=")

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]

    context_parts = []

    for index, (document, metadata) in enumerate(zip(documents, metadatas)):
        context_parts.append(
            f"DOCUMENT {index + 1}\n"
            f"SOURCE: {metadata['source']}\n"
            f"CONTENT:\n{document}"
        )

    context = "\n\n---\n\n".join(context_parts)

    
    prompt = textwrap.dedent(f"""
    You are answering a question using ONLY the provided documents.

    QUESTION:
    {query_text}

    DOCUMENTS:
    {context}

    Return ONLY valid JSON in exactly this format:

    {{
        "answer": "the complete value only",
        "citations": ["source/file/path.ext"],
        "confidence": 0.0
    }}

    Rules:

    1. Use ONLY information from the provided documents. Do not guess
       or use pretrained knowledge.

    2. Before selecting an answer, identify exactly what information
       the question requests and all conditions that must be satisfied.

    3. A value is a valid answer ONLY if the documents establish that
       it belongs to the exact item, condition, and context requested.

    4. Do not assume that a value associated with an item also applies
       to a specific condition that is not documented.

       For example:
       A price listed for TQ-40 does NOT establish its price at
       10,000-unit volume unless that volume-price relationship
       is explicitly supported by the documents.

    5. If even one required condition is unsupported, the question
       is unanswerable. Return:

       {{
           "answer": "",
           "citations": [],
           "confidence": 0.0
       }}

    6. Return the complete value requested, including necessary
       units, identifiers, revisions, and qualifiers.

    7. Do not select a value merely because it appears in a related
       document. It must actually answer the question.

    8. Cite the MINIMUM necessary set of files:
       - Include a file only if removing it makes the answer impossible
         to establish.
       - Include every file needed to connect evidence across documents.
       - Exclude related, duplicate, background, and unnecessary files.

    9. Prefer current and authoritative information over outdated,
       withdrawn, superseded, or deprecated information.

    10. Return ONLY the JSON object. Do not use Markdown code fences
        or explanations.
""").strip()

    
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            }
        ]
    }]

    model_result = vlm_pipe(text=messages)

    model_output = model_result[0]["generated_text"][-1]["content"].strip()

    if model_output.startswith("```"):
        model_output = model_output.removeprefix("```json").removeprefix("```")
        model_output = model_output.removesuffix("```").strip()

    print(f"model_output: {model_output}")
    response = json.loads(model_output)
    print(f"response: {response}")

    answer = str(response.get("answer", "")).strip()
    citations = response.get("citations", [])
    confidence = float(response.get("confidence", 0.0))
    
    validated_citations = validate_citations(
        query_text,
        answer,
        citations,
        documents,
        metadatas
    )
    
    if not answer:
        validated_citations = []
        confidence = 0.0
        
    final_response = {
        "answer": answer,
        "citations": validated_citations,
        "confidence": confidence
    }

    print(f"Final response: {final_response}")
    print(f"===Ending Resolving query Function===")

    return final_response


def request_handler(): 
    print("===STrating worker===")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen()

        print(f"Rag worker ready at {HOST}:{PORT}", flush=True)

        while True:
            connection, _ = server.accept()
            # handle_connection(connection, processor, model)
            with connection:
                    with connection.makefile("rwb") as stream:
                        try:
                            line = stream.readline()
                            if not line:
                                raise ValueError("Empty worker request.")
            
                            request = json.loads(line.decode("utf-8"))
                            # image_path = resolve_input_path(request["image_path"])
                            path = request.get("path")
                            actiion = request.get("action")
                            if path:
                                collection = find_covering_collection(path)
                            if not collection:
                                collection = process_index(path)

                            if actiion == "index":
                                response = {"message": f"Indexing completed for {path}"}
                            elif actiion == "query":
                                query_text = request.get("query_text", "")
                                if not query_text:
                                    raise ValueError("Missing 'query_text' for query action.")
                                response = resolve_query(collection, query_text)
                            else:
                                raise ValueError(f"Unknown action: {actiion}")
                            
                        except Exception as exc:
                            response = {"error": str(exc)}
            
                        stream.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
                        stream.flush()

request_handler()