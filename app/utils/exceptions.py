from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status
from app.utils.logger import get_logger

# This file catches errors in the FastAPI application and returns meaningful messages to the user.
# It handles validation errors and unexpected errors professionally.

logger = get_logger()


# This function is called when the user's data format is incorrect or wrong type
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        "Validation error on {method} {path}: {errors}",
        method=request.method,
        path=request.url.path,
        errors=exc.errors(),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Input validation failed",
            "errors": exc.errors(),
        },
    )


# This function is called when unexpected errors occur (file not found, model error)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on {method} {path}", method=request.method, path=request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


