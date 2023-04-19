FROM python:3.7-slim

RUN apt-get update
RUN apt-get -y install curl gnupg
RUN curl -sL https://deb.nodesource.com/setup_16.x  | bash -
RUN apt-get -y install nodejs

ENV CONTAINER_HOME=/var/www

ADD backend ${CONTAINER_HOME}/backend
ADD frontend ${CONTAINER_HOME}/frontend

WORKDIR ${CONTAINER_HOME}/frontend
RUN npm ci
RUN npm run build

WORKDIR ${CONTAINER_HOME}/backend

ARG DB_NAME
ENV DB_NAME $DB_NAME

RUN pip install -r ${CONTAINER_HOME}/backend/requirements.txt
