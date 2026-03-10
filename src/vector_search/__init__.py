from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel


# Based on https://gist.github.com/Aron-v1/f6e58554acf9ef0f328ac93d74dcb9ca
class SearchRequest(BaseModel):
    message: str

class SearchComponent(BaseModel):
    title: str
    url: str
    description: str
    parent: str
    accessability: str
    created_at: str
    updated_at: str
    has_research: bool
    views: int

class SearchResponse(BaseModel):
    message: str
    components: list[SearchComponent]

app = FastAPI()

instrumentator = Instrumentator().instrument(app)

@app.on_event("startup")
async def _startup():
    instrumentator.expose(app)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    return SearchResponse(message="Search successful", components=[])