FROM python:3.12-slim AS dev
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
RUN pip install --no-cache-dir -e ".[dev]"
CMD ["pytest", "tests/", "-v"]
