FROM nvcr.io/nvidia/pytorch:24.07-py3

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade --no-cache-dir pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . /app