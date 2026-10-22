
import os
import json
from pydantic import BaseModel, HttpUrl
from google import genai
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from imagekitio import ImageKit

load_dotenv()

app = FastAPI(title="TagIT Backend")
class WasteAnalysisRequest(BaseModel):
    image_url: HttpUrl





def get_imagekit_client():
    private_key = os.getenv("IMAGEKIT_PRIVATE_KEY")

    if not private_key:
        raise HTTPException(
            status_code=503,
            detail="ImageKit private key is missing.",
        )

    # The SDK uses https://api.imagekit.io by default.
    return ImageKit(private_key=private_key)





@app.get("/")
def home():
    return {"message": "TagIT backend is running!"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}



@app.get("/config-check")
def config_check():
    required = [
        "IMAGEKIT_PUBLIC_KEY",
        "IMAGEKIT_PRIVATE_KEY",
        "IMAGEKIT_URL_ENDPOINT",
    ]

    missing = [
        name for name in required
        if not os.getenv(name)
        or os.getenv(name).startswith("your_")
    ]

    if missing:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "ImageKit configuration is incomplete",
                "missing": missing,
            },
        )

    return {"imagekit_configured": True}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/jpg",
        "image/heic",
        "image/heif",
        "application/octet-stream",
    }
    content_type = (file.content_type or "").lower()

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image content type: {content_type}",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be 10 MB or smaller.",
        )

    try:
        client = get_imagekit_client()

        result = client.files.upload(
            file=BytesIO(image_bytes),
            file_name=file.filename or "waste-report.jpg",
            use_unique_file_name=True,
        )

        return {
            "message": "Image uploaded successfully",
            "image_url": result.url,
            "file_id": result.file_id,
        }

    except HTTPException:
        raise
    except Exception as error:
        print(f"ImageKit upload failed: {error}")
        raise HTTPException(
            status_code=502,
            detail="Image upload failed. Check backend logs.",
        ) from error
    finally:
        await file.close()


@app.get("/gemini-check")
async def gemini_check():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is missing from backend .env",
        )

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents="Reply with exactly: TagIT Gemini connection successful",
        )

        return {
            "status": "success",
            "message": response.text,
        }

    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API request failed: {type(e).__name__}: {e}",
        )



import httpx

@app.post("/analyze-waste")
async def analyze_waste(request: WasteAnalysisRequest):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is missing from backend .env",
        )

    try:
        # Download the image from its ImageKit URL.
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
        ) as http:
            image_response = await http.get(str(request.image_url))
            image_response.raise_for_status()

        image_bytes = image_response.content

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="The image URL returned an empty file.",
            )

        content_type = image_response.headers.get(
            "content-type", ""
        ).split(";")[0].lower()

        if content_type not in {
            "image/jpeg",
            "image/png",
            "image/webp",
        }:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image type: {content_type}",
            )

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[
                "You analyze images for TagIT, a civic waste-reporting app. "
                "Return JSON only with these fields: "
                "waste_detected (boolean), waste_type (string), "
                "description (string), severity (low, medium, high, or unknown). "
                "Describe only what is visibly supported by the image. "
                "If no waste is visible, set waste_detected to false "
                "and severity to unknown.",
                genai.types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=content_type,
                ),
            ],
            config={
                "response_mime_type": "application/json",
            },
        )

        import json
        analysis = json.loads(response.text)

        return {
            "status": "success",
            "analysis": analysis,
        }

    except HTTPException:
        raise
    except Exception as error:
        print(f"Waste analysis failed: {error}")
        raise HTTPException(
            status_code=502,
            detail="Waste analysis failed. Check backend logs.",
        ) from error

