FROM python:slim

RUN mkdir /work

WORKDIR /work

# COPY requirements.txt /work

RUN pip install progressbar \
numpy \
py_linq \
keras \
keras_metrics \
sklearn \
tensorflow

RUN apt -y update && apt -y install gcc

RUN pip install pyedflib

# RUN rm requirements.txt

COPY . /work

ENTRYPOINT bash