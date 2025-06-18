FROM python:3.11-slim

WORKDIR /app

# Copy dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh aplikasi
COPY . .

EXPOSE 80

# Jalankan Uvicorn via python -m untuk menghindari "executable not found"
CMD ["fastapi", "run", "/app/main.py", "--port", "80"]
