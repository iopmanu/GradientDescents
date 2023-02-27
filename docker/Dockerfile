FROM python:3.8-alpine

LABEL "project"="GradientDescents"
LABEL "creator"="iopmanu"

RUN apk add --no-cache --virtual .build-deps \
    build-base openssl-dev libffi-dev
RUN pip install numpy

COPY . /project

WORKDIR "/project"

CMD python3 main.py
