FROM tensorflow/tensorflow:2.6.0-gpu-jupyter

ENV PYTHONPATH "${PYTHONPATH}:/app/src/main/python"
ENV PYTHONPATH "${PYTHONPATH}:/app/src/main/scripts"
ENV TF_CPP_MIN_LOG_LEVEL 3
ENV AUTOGRAPH_VERBOSITY 0

WORKDIR /app

COPY . .

RUN mv src/main/scripts/71ssl-verify /etc/apt/apt.conf.d/71ssl-verify \
    && pip install -r src/main/python/requirements.txt