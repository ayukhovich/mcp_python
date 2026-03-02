FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY weather_server.py .

# Создаем директорию для логов
RUN mkdir -p /logs

# Запускаем сервер
CMD ["python", "-u", "weather_server.py"]