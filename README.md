# Moj Search

## Run Locally
```bash
uv run fastapi dev -e moj-search.main:app
```

## Build Local Image
```bash
podman build -f Dockerfile -t moj-search-api
```

## Run Local Image
```bash
podman run localhost/moj-search-api:latest
```


## TODO
[ ] - local model support (allow trust_remote_code=False)
[ ] - 1024 Dimension Size
[ ] - Why does deployment.yaml need to explicitely set the command?
[ ] - Deployment references image stream
[ ] - Multi-stage image build - libraries, application
[ ] - Metrics: LLM Hits (time to first token), Req/S
[ ] - Traces OLTP Traces (from MOJ)?
[ ] - Use OpenTelemetry https://github.com/open-telemetry/opentelemetry-python-contrib/tree/main/instrumentation/opentelemetry-instrumentation-fastapi
