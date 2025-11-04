from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Foam-Agent",
        "transport": os.environ.get("MCP_TRANSPORT", "stdio"),
    }