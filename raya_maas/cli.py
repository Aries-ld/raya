import logging

import uvicorn

from .app import create_app
from .config import Settings


def serve():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = Settings()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        workers=1,
        limit_concurrency=32,
        timeout_keep_alive=5,
    )


if __name__ == "__main__":
    serve()
