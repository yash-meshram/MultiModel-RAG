# MultiModel RAG
# pdf with images (text + images)

# will be use for loading teh pdf with image
import fitz     #PyMuPDF
from langchain_core.documents import Document
# for embedding
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

from langchain_core.messages import HumanMessage
import base64
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import typesense
from sentence_transformers import CrossEncoder

load_dotenv("./.env")

# Load CLIP mode which will be used for embedding
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# processor makes sure whatever input formate request for model it will try to convert in that format
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clip_model.eval()


# Embedding functions
def embed_image(image_data):
    """Embaded image using CLIP"""
    # if image path is given as image_data
    if isinstance(image_data, str):     #if path
        image = Image.open(image_data).convert("RGB")
    else:   #if PIL image
        image = image_data
        
    # below line will convert entire format into tensors
    inputs = clip_processor(images=image, return_tensors="pt")    # return in pytorch tensor
    
    with torch.no_grad():
        output = clip_model.get_image_features(**inputs)
        features = output.pooler_output          # Extract tensor directly
        # Normalizing Embedding into unit vectors
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

def embed_text(text):
    """embeded text using CLIP"""
    inputs = clip_processor(
        text=text,
        return_tensors = "pt",
        padding = True,
        truncation = True
    )
    
    with torch.no_grad():
        output = clip_model.get_text_features(**inputs)
        features = output.pooler_output          # Extract tensor directly
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()
    
def describe_image_with_llm(img_base64):
    """Send image to LLM and get description + extracted text"""
    
    message = HumanMessage(content=[
        {
            "type": "text",
            "text": """Analyze this image and provide:
            1. A detailed description of what you see
            2. Extract any text present in the image
            3. If it's a diagram/chart/table, explain its content
            
            Be as detailed as possible."""
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}",
                "detail": "high"
            }
        }
    ])
    
    response = llm.invoke([message])
    return response.content
    
# defining client for typsense search
client = typesense.Client({
    'nodes': [{
        'host': 'zqym3ix1p94j2r6vp-1.a2.typesense.net',  # For Typesense Cloud use xxx.a1.typesense.net
        'port': '443',       # For Typesense Cloud use 443
        'protocol': 'https'    # For Typesense Cloud use https
    }],
    'api_key': 'xxxxxxxxxxxxxxxxxxxxxx',
    'connection_timeout_seconds': 2
})

# Create collection schema
schema = {
    "name": "pdf_docs",
    "fields": [
        {"name": "id", "type": "string"},
        {"name": "content", "type": "string"},
        {"name": "page", "type": "int32"},
        {"name": "type", "type": "string", "facet": True},
        {"name": "image_id", "type": "string", "optional": True}
    ],
    "default_sorting_field": "page"
}
client.collections.create(schema)

# define LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Process pdf
pdf_path = "data/Attention is all you need.pdf"
document = fitz.open(pdf_path)

# store all documents and embedding and images
all_document = []
all_embedding = []
all_images = {}     # Store actual image data for LLM
all_images_doc = []
all_image_embedding = []

# text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)

for i, page in enumerate(document):
    # Process text
    page_text = page.get_text()
    
    if page_text.strip():
        temp_doc = Document(page_content=page_text, metadata={"page": i, "type": "text"})
        chunks = text_splitter.split_documents([temp_doc])
        
        # Embeded each chunk using CLIP
        for chunk in chunks:
            embedding = embed_text(chunk.page_content)
            all_embedding.append(embedding)
            all_document.append(chunk)
            
    # process image
    ##Three Important Actions:
    ##Convert PDF image to PIL format
    ##Store as base64 for GPT-4V (which needs base64 images)
    ##Create CLIP embedding for retrieval
    for img_index, img in enumerate(page.get_images(full = True)):
        try:
            xref = img[0]
            base_image = document.extract_image(xref)
            image_bytes = base_image["image"]
            
            # convert to PIL image
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # create unique id
            img_id = f"page_{i}_img_{img_index}"
            
            # store image as base64 for later use LLM
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            all_images[img_id] = img_base64
            
            # embed image using CLI
            embedding = embed_image(pil_image)
            all_embedding.append(embedding)
            # all_image_embedding.append(embedding)
            image_description = describe_image_with_llm(img_base64)
            embedding = embed_text(image_description)
            all_image_embedding.append(embedding)
            
            # create document for image
            image_doc = Document(
                # page_content = f"[Image: {img_id}]",
                page_content = image_description,
                metadata = {"page": i, "type": "image", "image_id": img_id}
            )
            all_document.append(image_doc)
            all_images_doc.append(image_doc)
            
        except Exception as e:
            print(f"Error processing image {img_index} on page {i}: {e}")
            continue
        
document.close()

all_document
all_embedding
all_images
all_images_doc
all_image_embedding

embedding_array = np.array(all_embedding)
# Create custome FAISS index since we have pre-computed embeddings
vector_store = FAISS.from_embeddings(
    text_embeddings = [(doc.page_content, emb) for doc, emb in zip(all_document, embedding_array)],
    # we are using precomputed embeddings thus
    embedding = None,
    metadatas = [doc.metadata for doc in all_document]
)

image_embedding_array = np.array(all_image_embedding)
vector_store_image = FAISS.from_embeddings(
    text_embeddings = [(doc.page_content, emb) for doc, emb in zip(all_images_doc, image_embedding_array)],
    # we are using precomputed embeddings thus
    embedding = None,
    metadatas = [doc.metadata for doc in all_images_doc]
)

# adding data in client collection
typesense_docs = []
for i, doc in enumerate(all_document):
    typesense_docs.append({
        'id': str(i),
        'content': doc.page_content,
        'page': doc.metadata.get('page', -1),
        'type': doc.metadata.get('type', 'text')
    })
for i, doc in enumerate(all_images_doc):
    typesense_docs.append({
        'id': str(i + len(all_document)),
        'content': doc.page_content,
        'page': doc.metadata.get('page', -1),
        'type': doc.metadata.get('type', 'text'),
        'image_id': doc.metadata.get('image_id', '-1')
    })
client.collections["pdf_docs"].documents.import_(typesense_docs)

# keyword search from typsense
def keyword_search(query, k = 5):
    search_parameters = {
        'q': query,
        'query_by': 'content',
        'per_page': k
    }
    
    results = client.collections["pdf_docs"].documents.search(search_parameters)
    
    hits = []
    for hit in results['hits']:
        hits.append({
            'content': hit['document']['content'],
            'page': hit['document']['page'],
            'type': hit['document']['type'],
            'image_id': hit['document'].get('image_id', None),
        })
    
    return hits


# reranker - will provide score - Good balance of speed + accuracy
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Retriever - Hybris search
def multimodel_retriever(query, k = 5):
    # 1. Vector Search
    # -----------------
    # Embeded query using CLIP
    query_embedding = embed_text(query)
    
    # Search in unified vector store
    results = vector_store.similarity_search_by_vector(
        embedding=query_embedding,
        k = k
    )
    results_ = vector_store_image.similarity_search_by_vector(
        embedding=query_embedding,
        k = k
    )
    vector_results = results + results_
    
    vector_hits = [{
        'doc': doc,
        'content': doc.page_content,
        'source': 'vector'
    } for doc in vector_results]
    
    # 2. Typesense search - search by keywords
    # ----------------------------------------
    keyword_hits_raw = keyword_search(query, k)
    
    keyword_hits = []
    for item in keyword_hits_raw:
        doc = Document(
            page_content = item['content'],
            metadata = {
                'page': item.get('page', -1),
                'type': item.get('type', 'text'),
                'image_id': item.get('image_id', '-1')
            }
        )
        
        keyword_hits.append({
            'doc': doc,
            'content': item['content'],
            'source': 'keyword'
        })
    
    # 3. Merge both retrieved result/hit + deduplicate
    # -------------------------------------------------
    combine_hits = vector_hits + keyword_hits
    
    seen = set()
    unique_results = []
    
    for item in combine_hits:
        if item['content'] not in seen:
            seen.add(item['content'])
            unique_results.append(item)
            
    # 4. Cross-Encoder reranking
    # ---------------------------
    pairs = [(query, item['content']) for item in unique_results]
    
    scores = reranker.predict(pairs)
    
    for i, item in enumerate(unique_results):
        item['score'] = float(scores[i])
        
    # 5. Sort
    # --------
    reranked_results = sorted(unique_results, key=lambda x: x['score'], reverse=True)
    
    # 6. Return DOcument
    # -------------------
    final_retrieved_docs = []
    
    for item in reranked_results[:k]:
        doc = item['doc']
        doc.metadata['score'] = item['score']
        doc.metadata['retrieved_from'] = item['source']
        
        final_retrieved_docs.append(doc)
    
    return final_retrieved_docs

def multimodel_prompt(query, retrieved_docs):
    content = []
    
    # adding instructions
    content.append(
        {
            "type": "text",
            "text": 
                """
                Amswer the query based on the provided text and image ONLY.
                Use only given information, dont use any other information even if asked to do so.
                Include image in your response if query is related to image.
                If you dont know the answer just say - 'No infrmation available'.
                """   
        }
    )
    
    # adding teh query
    content.append(
        {
            "type": "text",
            "text": query
        }
    )
    
    # seperate text and images
    retrieved_text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    retrieved_image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
    
    # add text doc to content
    if retrieved_text_docs:
        text_content = ""
        for doc in retrieved_text_docs:
            text_content += f"[Page {doc.metadata['page']}: {doc.page_content}]"
            content.append(
                {
                    "type": "text",
                    "text": text_content
                }
            )
    
    # add image doc to content
    if retrieved_image_docs:
        for doc in retrieved_image_docs:
            image_id = doc.metadata["image_id"]
            if image_id and image_id in all_images:
                content.append(
                    {
                        "type": "text",
                        "text": f"{doc.metadata['page']}",
                    }
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url":{
                            "url": f"data:image/png;base64,{all_images[image_id]}",
                            "detail": "high"
                            }
                    }
                )
        
    return HumanMessage(content = content)

# Enhanced query
def enhance_query(query):
    """Use LLM to enhance query for better retrieval"""
    
    message = HumanMessage(content=[
        {
            "type": "text",
            "text": f"""You are a query enhancement assistant.
            
                Your job is to rewrite the user query to make it more detailed and specific
                for retrieving relevant documents from a research paper.

                Original Query: {query}

                Rewrite the query by:
                1. Adding relevant technical terms
                2. Expanding abbreviations
                3. Including related concepts
                4. Being more specific and descriptive

                Return ONLY the enhanced query, nothing else. No explanation, no preamble."""
        }
    ])
    
    response = llm.invoke([message])
    return response.content

# building the pipeline
def multimodel_pdf_rag_pipeline(query_):
    # Enhanced the query
    query = enhance_query(query_)
    
    # Retrieve relevant documents
    retrieved_docs = multimodel_retriever(query = query, k = 5)    
    
    # cretae prompt
    prompt = multimodel_prompt(query = query, retrieved_docs = retrieved_docs)
    
    # get response
    response = llm.invoke([prompt])
    
    if response.content != 'No information available.':
        retrieved_image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
        seen_images = set()
        for doc in retrieved_image_docs:
            image_id = doc.metadata["image_id"]
            if image_id and image_id in all_images and image_id not in seen_images:
                seen_images.add(image_id)
                image_bytes = base64.b64decode(all_images[image_id])
                image = Image.open(io.BytesIO(image_bytes))
                image.show()  
    
    return response.content, retrieved_image_docs

if __name__ == "__main__":
    query = "Explain transformer each and every steps with images"
    response, retrieved_image_docs = multimodel_pdf_rag_pipeline(query)
    print(response)
    print(retrieved_image_docs)