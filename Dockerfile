FROM nvcr.io/nvidia/tensorflow:21.11-tf2-py3

ENV PYTHONPATH "${PYTHONPATH}:/app/src/main/python"
ENV PYTHONPATH "${PYTHONPATH}:/app/src/main/scripts"
ENV TF_CPP_MIN_LOG_LEVEL 3
ENV AUTOGRAPH_VERBOSITY 0

WORKDIR /app

COPY . .

RUN pip install -r src/main/python/requirements.txt