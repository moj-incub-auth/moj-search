# Moj Search

## Run Locally
```bash
uv run fastapi dev -e moj-search.main:app
```

## Build Local Image
```bash
podman build -f Dockerfile -t moj-search
```

## Run Local Image
```bash
podman run localhost/moj-search:latest
```


## TODO
[ ] - local model support (allow trust_remote_code=False)
