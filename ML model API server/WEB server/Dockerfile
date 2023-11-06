FROM python:3

ENV PYTHONUNBUFFERED 1

WORKDIR /app

ADD . /app

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update --fix-missing && apt-get install -y python3-opencv

RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . /app