FROM python:slim

RUN mkdir /work

WORKDIR /work

# COPY requirements.txt /work

RUN pip install progressbar \
numpy \
py_linq \
keras \
keras_metrics \
tensorflow

RUN apt -y update && apt -y install gcc

RUN pip install pyedflib

# RUN rm requirements.txt

COPY . /work

EXPOSE 6006

ENTRYPOINT ["tensorboard", "--host", "0.0.0.0", "--port", "6006", "--logdir", "logs"]