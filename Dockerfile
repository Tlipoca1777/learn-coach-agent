FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps -e .

COPY app.py ./app.py

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "memory_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
