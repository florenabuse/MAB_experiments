FROM pytorch/pytorch

WORKDIR ./

COPY ./classification-based/requirements.txt ./
RUN pip3 install -r requirements.txt

COPY .. .