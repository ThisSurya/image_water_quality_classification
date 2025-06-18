# make a class for the model
from typing import Any, Dict, List
from pydantic import BaseModel, Field
import tensorflow as tf
import numpy as np
import io
from typing import Union
from PIL import Image
MODEL_PATH = "models/mobilenet.h5"
IMAGE_TARGET_SIZE = (224, 224)
class ImageClassifier:
    def __init__(self, model_path: str, class_names: List[str]):
        """
        Konstruktor untuk kelas ImageClassifier.
        Args:
            model_path (str): Path menuju file model .h5.
            class_names (List[str]): Daftar nama kelas.
        """
        self.model_path = model_path
        self.class_names = class_names
        self.model = self._load_model()

    def _load_model(self) -> tf.keras.Model:
        """
        Metode internal untuk memuat model dari path yang diberikan.
        """
        try:
            model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model dari '{self.model_path}' berhasil dimuat.")
            return model
        except Exception as e:
            print(f"❌ Error saat memuat model: {e}")
            return None

    @staticmethod
    def preprocess_image(image_bytes: bytes, target_size: tuple) -> np.ndarray:
        """
        Metode statis untuk melakukan preprocessing pada gambar.
        Tidak memerlukan 'self' karena logikanya independen.
        
        PENTING: Sesuaikan langkah-langkah ini agar SAMA PERSIS dengan 
        preprocessing saat training model Anda.
        """
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(target_size)
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = image_array / 255.0  # Normalisasi
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def predict(self, image_bytes: bytes) -> Dict[str, Union[str, float]]:
        """
        Melakukan prediksi pada byte gambar yang diberikan.
        """
        if self.model is None:
            raise RuntimeError("Model belum dimuat atau gagal dimuat.")

        # 1. Preprocess gambar
        processed_image = self.preprocess_image(image_bytes, IMAGE_TARGET_SIZE)

        # 2. Lakukan prediksi
        raw_prediction = self.model.predict(processed_image)
        
        # 3. Proses hasil prediksi
        score = float(np.max(raw_prediction))
        predicted_class_index = np.argmax(raw_prediction)
        predicted_class_name = self.class_names[predicted_class_index]
        
        return {
            "predicted_class": raw_prediction.tolist(),
            "confidence_score": score
        }