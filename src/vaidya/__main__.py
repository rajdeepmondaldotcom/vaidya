"""Entry point for `python -m vaidya`."""

import uvicorn

from vaidya.config import Settings


def main() -> None:
    settings = Settings()
    uvicorn.run(
        "vaidya.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
