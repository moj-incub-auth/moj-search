from pydantic import BaseModel

# Based on https://gist.github.com/Aron-v1/f6e58554acf9ef0f328ac93d74dcb9ca
class SearchRequest(BaseModel):
    message: str

class SearchComponent(BaseModel):
    title: str
    url: str
    description: str
    parent: str
    accessibility: str
    created_at: str | None
    updated_at: str | None
    has_research: bool
    views: int

class ScoredSearchComponent(SearchComponent):
    score: float

class SearchResponse(BaseModel):
    message: str
    components: list[SearchComponent]