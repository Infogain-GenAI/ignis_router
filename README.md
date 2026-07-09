# ignis_router

Reusable LLM routing library with API and CLI examples.

## Run API (Default Port)

Use the built-in runner command:

```powershell
python -m ignis_router.run_api
```

Defaults:

- Host: `127.0.0.1`
- Port: `8013`
- Reload: `false`
- If API is already running on `8013`, the runner exits cleanly with a message.
- If another process owns `8013`, the runner fails with a clear error.

Override using environment variables:

- `API_PORT` or `IGNIS_ROUTER_API_PORT`
- `IGNIS_ROUTER_API_RELOAD` (`true`/`false`)

## API Endpoints

- `GET /`
- `GET /health`
- `POST /route`
- `GET /docs`

## Quick Test

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8013/route" -Method Post -ContentType "application/json" -Body '{"query":"Generate Python code for data analysis"}'
```
