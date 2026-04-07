import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from supabase import create_client
from app.services.pdf_reader import PDFReader
from app.services.documind import (
    chunk_and_store,
    query,
    create_session,
    save_message,
    get_all_messages,
    list_sessions,
    answer_question,
    extract_video_id,
    get_transcript
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "SourceMind is running successfully !"}


@app.post('/upload_pdf')
def upload(user_id: str = Form(...), files: List[UploadFile] = File(...)):
    for file in files:
        pdf_bytes = file.file.read()
        raw_text = PDFReader(pdf_bytes, from_bytes=True)
        chunk_and_store(raw_text, file.filename, user_id)
    return {"message": "Files uploaded successfully"}


class ChatRequestYoutubeLink(BaseModel):
    url: str
    user_id: str


@app.post('/upload_youtube_link')
def upload_youtube(req: ChatRequestYoutubeLink):
    video_id = extract_video_id(req.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    try:
        raw_text = get_transcript(video_id)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    chunk_and_store(raw_text, video_id, req.user_id)
    return {
        "message": "Transcript indexed successfully",
        "doc_name": video_id
    }


class NewSession(BaseModel):
    doc_names: List[str]
    title: str
    user_id: str


@app.post('/session/new')
def new_session(session: NewSession):
    session_id = create_session(session.doc_names, session.title, session.user_id)
    return {"session_id": session_id}


@app.get('/sessions')
def get_sessions(user_id: str):
    return list_sessions(user_id)


@app.get('/session/{session_id}/messages')
def messages(session_id: str):
    return get_all_messages(session_id)


class ChatRequestPDF(BaseModel):
    session_id: str
    query_text: str
    doc_names: List[str]
    user_id: str


@app.post('/chat_pdfs')
def chat_pdf(req: ChatRequestPDF):
    reply = answer_question(
        req.session_id,
        req.query_text,
        req.doc_names,
        req.user_id
    )
    return {"reply": reply}


class ChatRequestYoutube(BaseModel):
    session_id: str
    query_text: str
    doc_names: List[str]
    user_id: str


@app.post('/chat_youtube')
def chat_youtube(req: ChatRequestYoutube):
    reply = answer_question(
        req.session_id,
        req.query_text,
        req.doc_names,
        req.user_id
    )
    return {"reply": reply}


#delete
@app.delete('/session/{session_id}')
def delete_session(session_id: str):
    if not session_id or session_id == 'null':
        return {"status": "skipped"}
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    # Delete messages first (foreign key), then session
    supabase.table("chat_messages").delete().eq("session_id", session_id).execute()
    supabase.table("chat_sessions").delete().eq("id", session_id).execute()
    return {"status": "deleted"}