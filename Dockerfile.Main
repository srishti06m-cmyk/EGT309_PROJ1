
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

WORKDIR /project

COPY pyproject.toml README.md requirements.txt ./
RUN pip install --upgrade pip && \
	pip install -e .

COPY . .

RUN chmod +x run.sh

ENTRYPOINT ["./run.sh"]