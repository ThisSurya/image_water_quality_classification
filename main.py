# load model
import os
import sys
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ImageClassifier import ImageClassifier
app = FastAPI()
# Mengizinkan Cross-Origin Resource Sharing (CORS) agar bisa diakses dari frontend yang berbeda domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Atau spesifikasikan domain frontend Anda, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variabel global untuk menampung objek classifier kita
# Type hint 'ImageClassifier' membantu editor kode untuk auto-completion
classifier: ImageClassifier = None
MODEL_PATH = "model/mobilenet.keras"
CLASS_NAMES = ["Lainnya", "Kotor", "Bersih"]  # Sesuaikan dengan kelas yang Anda miliki
@app.on_event("startup")
async def startup_event():
    """
    Saat aplikasi dimulai, buat satu instance dari ImageClassifier.
    Objek ini akan digunakan kembali untuk semua request.
    """
    global classifier
    print("🚀 Aplikasi FastAPI sedang dimulai... Membuat instance classifier.")
    classifier = ImageClassifier(MODEL_PATH, CLASS_NAMES)

@app.get("/", tags=["General"])
def read_root():
    return {"message": "Selamat datang di API Prediksi Gambar v2.0 (Struktur Kelas)!"}

@app.post("/predict", tags=["Prediction"])
async def predict_image_endpoint(file: UploadFile = File(...)):
    """
    Logika utama diserahkan kepada objek 'classifier'.
    """
    if classifier is None or classifier.model is None:
        return JSONResponse(status_code=503, content={"error": "Model tidak tersedia atau gagal dimuat. Silakan cek log server."})

    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "File yang diupload harus berupa gambar."})

    try:
        image_bytes = await file.read()
        # image_preprocessed = classifier.preprocess_image(image_bytes, target_size=(160, 160), model_type="mobilenet")
        
        # Cukup panggil metode predict dari objek classifier
        result = classifier.predict(image_bytes)
        
        # Format ulang confidence score untuk response
        result['confidence_score'] = f"{result['confidence_score']:.2%}"
        
        return JSONResponse(status_code=200, content=result)
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Terjadi kesalahan: {str(e)}"})