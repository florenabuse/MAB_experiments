# syntax=docker/dockerfile:1

FROM python:3.11

WORKDIR ./

# RUN pip install -U pip
# RUN pip install --upgrade setuptools
# RUN apt-get install sudo
# RUN sudo apt-get install python3-setuptools

COPY ./classification-based/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY .. .

CMD [ "python", "./MAB_main.py" ]