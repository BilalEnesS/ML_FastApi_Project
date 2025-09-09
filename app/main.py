from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from app.utils.logger import configure_logger
from app.utils.exceptions import (
    validation_exception_handler,
    unhandled_exception_handler,
)
from app.routes.upload import router as upload_router
from app.routes.config import router as config_router
from app.routes.train import router as train_router
from app.routes.predict import router as predict_router
from app.routes.metrics import router as metrics_router

# This file is the main entry point for the FastAPI application.
# It brings together all routes, exception handlers and static files.
# It serves HTML/CSS/JS files for the user interface.

configure_logger()

# Create the FastAPI application
app = FastAPI()

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Show user interface
@app.get("/")
def read_root():
    return FileResponse("app/static/index.html")

# Error management 
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# API routes
app.include_router(upload_router)      # /upload 
app.include_router(config_router)      # /config 
app.include_router(train_router)       # /train 
app.include_router(predict_router)     # /predict 
app.include_router(metrics_router)     # /metrics
