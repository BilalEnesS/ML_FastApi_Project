from __future__ import annotations
from io import BytesIO
from pathlib import Path
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette import status
from app.utils.logger import get_logger


# Excel file upload and CSV conversion
router = APIRouter(tags=["Upload"])
logger = get_logger()


@router.post("/upload")
async def upload_excel(file: UploadFile = File(...)):
    filename = (file.filename or "").strip()
    if not filename.lower().endswith(".xlsx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .xlsx files are accepted",
        )

    try:
        # Read the file content as bytes and convert it to a DataFrame in memory
        content = await file.read()
        df = pd.read_excel(BytesIO(content), engine="openpyxl")

        # If the data directory does not exist, create it and save it as CSV
        data_dir = Path("app/data")
        data_dir.mkdir(parents=True, exist_ok=True)
        output_path = data_dir / "temp_data.csv"
        df.to_csv(output_path, index=False)

        logger.info(
            "Uploaded file saved to {path} with {rows} rows and {cols} columns",
            path=str(output_path),
            rows=df.shape[0],
            cols=df.shape[1],
        )

        return {
            "message": "File processed and saved",
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "saved_path": str(output_path),
        }
    except HTTPException:
        raise
    except Exception as exc:
        # If there is an error during reading/conversion, return a meaningful response to the user
        logger.exception("Upload processing failed: {error}", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to process the uploaded .xlsx file",
        )


