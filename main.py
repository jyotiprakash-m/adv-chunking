"""
main.py — Entry point for the Advanced RAG Backend (PDF Chunking API)

This file initializes the FastAPI app, includes the /chunk router,
and configures CORS so your Next.js frontend can call it freely.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.chunk_router import router as chunk_router
from routers.career_center_router import router as career_center_router
from routers.deep_research_router import router as deep_research_router

# =========================================
# ✅ Initialize FastAPI app
# =========================================
app = FastAPI(
    title="Generative Artificial Intelligence API",
    description="Api backend for various AI-powered functionalities",
    version="1.0.0",
)

# =========================================
# 🌐 CORS Middleware (for Next.js frontend)
# =========================================
# You can update `origins` later to your frontend’s actual domain
origins = [
    "http://localhost:3000",
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
app.include_router(career_center_router)
app.include_router(deep_research_router)

# =========================================
# 🏁 Root endpoint
# =========================================
@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Welcome to the Generative Artificial Intelligence API 🚀",
        "endpoints": {
            "POST /chunk": "Upload a PDF and get its text chunks.",
            "POST /deep-research/run": "Perform deep research on a query and get a detailed report.",
            "POST /career/chat": "Get in touch via the Career Center chatbot."
        }
    }

# =========================================
# 🖥️ Run with: uvicorn main:app --reload
# =========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# uv run uvicorn main:app --reload