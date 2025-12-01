# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ml.inference import predict_image_bytes

app = FastAPI(
    title="Plant Classifier",
    description="Upload a leaf image and get disease predictions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Plant Pathology API is running. Go to /docs for Swagger UI."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()

    result = predict_image_bytes(image_bytes)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        **result,
    }
