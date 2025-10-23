"""
main.py — Entry point for the Advanced RAG Backend (PDF Chunking API)

This file initializes the FastAPI app, includes the /chunk router,
and configures CORS so your Next.js frontend can call it freely.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.chunk_router import router as chunk_router

# =========================================
# ✅ Initialize FastAPI app
# =========================================
app = FastAPI(
    title="PDF Chunking & RAG API",
    description="API backend for document chunking, processing, and RAG workflows",
    version="1.0.0",
)

# =========================================
# 🌐 CORS Middleware (for Next.js frontend)
# =========================================
# You can update `origins` later to your frontend’s actual domain
origins = [
    "http://localhost:3000",  # Next.js local dev
    "https://jyoti-prakash-mohanta.vercel.app",  # production frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # or ["*"] for all (less secure)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# 📦 Include Routers
# =========================================
app.include_router(chunk_router)

# =========================================
# 🏁 Root endpoint
# =========================================
@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Welcome to the PDF Chunking & RAG API 🚀",
        "endpoints": {
            "POST /chunk": "Upload a PDF and get its text chunks.",
        }
    }

# =========================================
# 🖥️ Run with: uvicorn main:app --reload
# =========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# uv run uvicorn main:app --reload