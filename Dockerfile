FROM python:3.11-slim

WORKDIR /app

# Copy application files
COPY . /app

# Railway provides PORT dynamically
ENV PORT=8080
EXPOSE 8080

CMD ["python", "server.py"]
