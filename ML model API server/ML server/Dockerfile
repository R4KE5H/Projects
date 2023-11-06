FROM python:3.7.13

ENV PYTHONUNBUFFERED 1

WORKDIR /flaskapp

ADD . /flaskapp

COPY ./requirements.txt /flaskapp/requirements.txt

RUN apt-get update --fix-missing && apt-get install -y python3-opencv

RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . /flaskapp