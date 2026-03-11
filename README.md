# Vector Search

## Run Locally
```bash
uv run fastapi dev src/vector_search
```


## Build Local Image
```bash
podman build -f Dockerfile -t vector_search
```

## Run Local Image
```bash
podman run localhost/vector-search:latest
```


## TODO
[x] - pymilvus support
[ ] - fastapi - lifecycle migration
[x] - kserve support
[x] - local sentence-transformers fallback
[ ] - services wiring
[ ] - local model support (allow trust_remote_code=False)
