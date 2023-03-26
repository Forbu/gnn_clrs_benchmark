# start from cuda 11.7 image
FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04
RUN apt update -y

# RUN apt install software-properties-common -y
RUN apt install -y python3-pip -y

# install pytorch
RUN pip3 install torch torchvision torchaudio

# set working directorye
WORKDIR /app

RUN pip install --upgrade pip
# add requirements.txt

COPY requirements.txt .
# install python dependencies (requirements.txt)
RUN pip install -r requirements.txt
