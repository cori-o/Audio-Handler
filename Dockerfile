# Dockerfile for Audio-Handler  
# write by Jaedong, Oh (2025.03.26)
# you should replace BASE_IMAGE to your local env 
ARG BASE_IMAGE=pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
FROM ${BASE_IMAGE}
WORKDIR /audio-handler 
COPY . .
ENV LC_ALL=ko_KR.UTF-8 

RUN apt-get update && apt-get install -y locales
RUN locale-gen ko_KR.UTF-8   
RUN apt-get install python3-pip -y
RUN apt-get install vim -y 
RUN apt-get update && apt-get install git -y
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt