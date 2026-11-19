from fastapi import APIRouter, UploadFile, File, HTTPException
import logging
from pathlib import Path
from ..utils.file_handler import save_uploaded_file, ALLOWED_REFERENCE_EXTENSIONS, validate_file_extension
from ..services.reference_service import process_reference_document

router = APIRouter(prefix="/api/upload", tags=["reference"])
logger = logging.getLogger(__name__)

@router.post("/reference")
async def upload_reference_key(file: UploadFile = File(...)):
    """
    Accepts reference answer key (PDF, TXT, DOCX).
    Saves the file, extracts text, parses questions, reference answers, and marking rubrics (Pipeline 2),
    and returns the structured QA bank.
    """
    if not validate_file_extension(file.filename, ALLOWED_REFERENCE_EXTENSIONS):
        ext = Path(file.filename).suffix.lower()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Allowed types: {list(ALLOWED_REFERENCE_EXTENSIONS)}"
        )
    
    try:
        saved_info = await save_uploaded_file(file, subfolder="references")
        result = process_reference_document(saved_info["stored_path"], saved_info["file_id"])
        result["filename"] = file.filename
        result["file_id"] = saved_info["file_id"]
        result["size_formatted"] = saved_info["size_formatted"]
        return result
    except Exception as e:
        logger.error(f"Error processing reference key: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process reference answer key: {str(e)}")
