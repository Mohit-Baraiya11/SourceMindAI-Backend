import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from groq import Groq
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec
from langchain_experimental.text_splitter import SemanticChunker
from supabase import create_client
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import re

load_dotenv()


def get_embedding_model():
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
    )

def get_pinecone_index():
    return "documind-ai"


def create_pinecone_vector_store():
    index_name = get_pinecone_index()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    existing = [i["name"] for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='dotproduct',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    return pc.Index(index_name)


# ------------------- CHUNK + STORE -------------------
def chunk_and_store(raw_text: str, doc_name: str, user_id: str):
    if not raw_text or not raw_text.strip():
        print(f"'{doc_name}' has no extractable text — skipping.")
        return

    raw_text = raw_text.strip()
    raw_text = re.sub(r'[^\x00-\x7F]+', ' ', raw_text)
    raw_text = re.sub(r'\n{3,}', '\n\n', raw_text)
    raw_text = re.sub(r' {2,}', ' ', raw_text)

    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    # prevent duplicate indexing per user
    existing = supabase.table("parent_chunks")\
        .select("id")\
        .eq("doc_name", doc_name)\
        .eq("user_id", user_id)\
        .limit(1)\
        .execute()

    if existing.data:
        print(f"'{doc_name}' already indexed for this user — skipping.")
        return

    embedding_model = get_embedding_model()

    parent_splitter = SemanticChunker(embedding_model, breakpoint_threshold_amount=95)
    child_splitter = SemanticChunker(embedding_model, breakpoint_threshold_amount=70)

    index = create_pinecone_vector_store()
    parent_chunks = parent_splitter.split_text(raw_text)

    all_children = []
    parent_map = []

    for i, parent_text in enumerate(parent_chunks):
        parent_id = f"{doc_name}_parent_{i}"

        # ✅ store with user_id
        supabase.table("parent_chunks").upsert({
            "id": parent_id,
            "content": parent_text,
            "doc_name": doc_name,
            "user_id": user_id
        }).execute()

        child_chunks = child_splitter.split_text(parent_text)

        for j, child_chunk in enumerate(child_chunks):
            all_children.append(child_chunk)
            parent_map.append((f"{parent_id}_child_{j}", parent_id))

    bm25 = BM25Encoder()
    bm25.fit(all_children)
    supabase.table("bm25_params").upsert({
        "id": f"{doc_name}_{user_id}",
        "doc_name": doc_name,
        "user_id": user_id,
        "params": bm25.get_params()
    }).execute()

    sparse_vectors = bm25.encode_documents(all_children)

    vectors = []
    for idx, (child_text, sparse) in enumerate(zip(all_children, sparse_vectors)):
        child_id, parent_id = parent_map[idx]

        if not sparse.get('values') or len(sparse['values']) == 0:
            continue

        if not child_text or len(child_text.strip()) < 10:
            continue

        dense = embedding_model.embed_query(child_text)

        vectors.append({
            "id": child_id,
            "values": dense,
            "sparse_values": sparse,
            "metadata": {
                "parent_id": parent_id,
                "text": child_text,
                "doc_name": doc_name,
                "user_id": user_id   # ✅ added
            }
        })

    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i+100])

    print(f"Done — stored for user {user_id}")


# ------------------- QUERY -------------------
def query(query_text: str, doc_name: str, user_id: str):
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    embedding_model = get_embedding_model()
    index = create_pinecone_vector_store()

    result_bm25 = supabase.table("bm25_params")\
        .select("params")\
        .eq("id", f"{doc_name}_{user_id}")\
        .execute()
    params = result_bm25.data[0]["params"]
    bm25 = BM25Encoder()
    bm25.__dict__.update(params)

    dense_vector = embedding_model.embed_query(query_text)
    sparse_vector = bm25.encode_queries([query_text])[0]

    results = index.query(
        vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=10,
        include_metadata=True,
        filter={
            "doc_name": {"$eq": doc_name},
            "user_id": {"$eq": user_id}
        }
    )

    parent_ids = list(set([m["metadata"]["parent_id"] for m in results["matches"]]))

    response = supabase.table("parent_chunks")\
        .select("content")\
        .in_("id", parent_ids)\
        .eq("user_id", user_id)\
        .execute()

    context = "\n\n".join([item["content"] for item in response.data])
    return context


# ------------------- SESSION -------------------
def create_session(doc_names, title, user_id):
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    result = supabase.table("chat_sessions").insert({
        "doc_names": doc_names,
        "title": title,
        "user_id": user_id
    }).execute()

    return result.data[0]["id"]


def save_message(session_id, role, content, user_id):
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    supabase.table("chat_messages").insert({
        "session_id": session_id,
        "role": role,
        "content": content,
        "user_id": user_id
    }).execute()


def get_last10_messages(session_id):
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    result = supabase.table("chat_messages")\
        .select("role","content")\
        .eq("session_id", session_id)\
        .order("created_at", desc=True)\
        .limit(10)\
        .execute()

    return list(reversed(result.data))


def get_all_messages(session_id):
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    result = supabase.table("chat_messages")\
        .select("role","content")\
        .eq("session_id", session_id)\
        .order("created_at", desc=True)\
        .execute()

    return list(reversed(result.data))


def list_sessions(user_id):
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    result = supabase.table("chat_sessions")\
        .select("id","title","doc_names","created_at")\
        .eq("user_id", user_id)\
        .order("created_at", desc=True)\
        .execute()

    return result.data


# ------------------- ANSWER -------------------
def answer_question(session_id, query_text, doc_names, user_id):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    messages = get_last10_messages(session_id)

    context = ""
    for doc_name in doc_names:
        context += f"\n\nFrom {doc_name}:\n{query(query_text, doc_name, user_id)}"

    answer = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": f"""You are a helpful assistant. The user has provided source material which may be from a PDF document or a YouTube video transcript. Answer the user's questions based strictly on this source material. Never say you cannot access videos, URLs, or external content — the content has already been extracted and provided to you as text below. If the answer is not in the provided content, say 'This information is not available in the provided source.'
            Source content:
            {context}"""},
            *messages,
            {"role": "user", "content": query_text}
        ],
        max_tokens=1024,
    )

    reply = answer.choices[0].message.content

    save_message(session_id, "user", query_text, user_id)
    save_message(session_id, "assistant", reply, user_id)

    return reply


# ------------------- YOUTUBE -------------------
def extract_video_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]+)",
        r"youtu\.be/([a-zA-Z0-9_-]+)",
        r"shorts/([a-zA-Z0-9_-]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def get_transcript(video_id):
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=['hi', 'en'])
        text = " ".join([t.text for t in transcript])
        if not text or len(text.strip()) < 50:
            raise ValueError("Transcript too short or empty")
        return text
    except Exception as e:
        raise RuntimeError(f"Could not fetch transcript: {str(e)}")