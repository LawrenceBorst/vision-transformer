FROM ubuntu:25.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get install -y python3-pip

RUN pip install poetry --break-system-packages

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml poetry.lock README.md ./

RUN poetry env use python3
RUN poetry install --no-root

COPY /model/. /app/model
COPY /static/. /app/static
COPY /src/. /app/src
COPY /scripts/. /app/scripts

ENV PYTHONPATH="${PYTHONPATH}:/app" \
    VIRTUAL_ENV=/app/.venv \
    PATH="$VIRTUAL_ENV/bin:/root/.local/bin:$PATH"
