FROM golang:1.25.2-bookworm AS go-builder

ENV GOPATH=/go
ENV PATH=$GOPATH/bin:$PATH

FROM go-builder AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    make \
    bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /go/src

ARG TARGETOS=linux
ARG TARGETARCH=amd64

ENV GOOS=linux
ENV GOARCH=amd64

COPY go.mod go.sum ./
RUN go mod download
COPY . .

RUN echo "Building for ${GOOS}/${GOARCH}" && \
    CGO_ENABLED=1 go build -o orus-api .

# Stage final
FROM debian:bookworm-slim

WORKDIR /go/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3 \
    python3-pip \
    python3-venv \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-por \
    tesseract-ocr-spa \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    postgresql-client \
    wget \
    curl \
    ca-certificates \
    opus-tools \
    lame \
    libvorbis0a \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN python3 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir \
        PyPDF2 pdfplumber pytesseract pillow \
        pgai psycopg2-binary asyncpg \
        pydub speechrecognition ffmpeg-python \
        numpy gdown

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
ENV LD_LIBRARY_PATH=/go/bin/onnx/aarch64:$LD_LIBRARY_PATH

COPY --from=build /go/src/orus-api .

COPY .env /go/bin/.env

RUN chmod 644 /go/bin/.env

RUN mkdir -p  /go/bin/onnx

COPY ./onnx/. /go/bin/onnx/

RUN useradd -u 799 -m orus-api && \
    chown -R orus-api:orus-api /go/bin && \
    chmod +x /go/bin/orus-api

USER orus-api

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD pgrep orus-api || exit 1

LABEL maintainer="Dsouza10082" \
      version="1.0.0" \
      description="Orus API with Go 1.25.2, FFmpeg, and ONNX support"

ENTRYPOINT ["/go/bin/orus-api"]
CMD []