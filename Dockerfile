FROM python:3.10.6-buster

# Install HDF5 and libGL for OpenCV
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libgl1 \
    && apt-get clean

COPY fast_api fast_api
COPY requirements.txt requirements.txt
COPY classification classification
COPY segmentation segmentation
COPY models models

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn fast_api.api:app --host 0.0.0.0 --port $PORT
