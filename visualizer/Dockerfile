FROM python:slim

RUN pip install tensorboard

RUN mkdir /work

WORKDIR /work

RUN mkdir logs

EXPOSE 6006

CMD ["tensorboard", "--host", "0.0.0.0", "--port", "6006", "--logdir", "logs"]