"""Authenticated local HTTP service for time-series forecasting and scoring."""

import hmac
import json
import os
import threading

MAX_BODY = 4 << 20


def create_app(engine):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    token = os.environ.get("TIMESERIES_API_KEY", "")
    if not token:
        raise ValueError("TIMESERIES_API_KEY is required")
    lock = threading.Lock()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/timeseries/{operation}")
    async def infer(operation: str, request: Request):
        if not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if operation not in ("forecast", "anomalies"):
            return JSONResponse({"error": "unknown operation"}, status_code=404)
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > MAX_BODY:
                return JSONResponse({"error": "request too large"}, status_code=413)
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeError):
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        if not lock.acquire(blocking=False):
            return JSONResponse({"error": "model is busy"}, status_code=429)
        try:
            from starlette.concurrency import run_in_threadpool
            result = await run_in_threadpool(engine.run, body, operation)
            response = JSONResponse(result)
            if len(response.body) > MAX_BODY:
                return JSONResponse({"error": "response exceeds size limit; reduce the inputs or horizon"}, status_code=413)
            return response
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception:
            # Never include submitted series or provider traces in HTTP errors.
            return JSONResponse({"error": "model inference failed"}, status_code=500)
        finally:
            lock.release()

    return app


def main(argv=None):
    import argparse
    from model import MODEL_NAMES, load_engine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default="timesfm-3.0",
                        help="checkpoint to serve (default: %(default)s)")
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(create_app(load_engine(args.model)),
                host=os.environ.get("TIMESERIES_HOST", "127.0.0.1"), port=8091)


if __name__ == "__main__":
    main()
