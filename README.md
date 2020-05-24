# Предикт эпизодов эпилепсии

Написали на Keras, и оно, вроде, даже немножко работает.

## Развертывание системы через контейнеры Docker

С целью упрощения подготовки окружения и запуска скриптов мы собрали образ для Docker, внутри которого можно запускать нашу модель.

Для сборки и запуска используем следующие команды:

    > git clone https://github.com/kav128/SkyNet
    > docker build -t skynet skynet
    > docker run --rm -dp 6006:6006 --name skynet skynet

Контейнер запущен, по адресу `localhost:6006` доступен Tensorboard.

Для запуска скрипта используем команду:

    > docker exec -it skynet python <script_name>.py

## Локальный запуск скриптов на Python

Если не хотим использовать Docker, можно также запустить скрипты локально. Все зависимости перечислены ниже:

- progressbar
- numpy
- py_linq
- pyedflib
- keras
- keras_metrics
- sklearn
- tensorflow

Также все они прописаны в  `requrements.txt`

    > git clone https://github.com/kav128/SkyNet
    > pip install -r requrements.txt
    > cd SkyNet
    SkyNet> python <script_name>.py

Также в папку SkyNet/logs записываются логи для Tensorboard, поэтому:

    SkyNet> tensorboard --logdir logs
