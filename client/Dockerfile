FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mcp_client.py .
COPY weather_server.py .  # Копируем сервер для запуска

# Создаем директорию для логов
RUN mkdir -p /logs

# Запускаем клиент в интерактивном режиме
CMD ["python", "-u", "mcp_client.py"]