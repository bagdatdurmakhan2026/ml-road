# 1. Базовый образ Python
FROM python:3.11-slim

# 2. Рабочая директория внутри контейнера
WORKDIR /app

# 3. Копируем и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Копируем исходный код и обученную модель
COPY app.py train.py model.pkl ./

# 5. Открываем порт FastAPI
EXPOSE 8000

# 6. Команда запуска сервиса
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]