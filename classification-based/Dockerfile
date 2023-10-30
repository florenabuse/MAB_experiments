FROM pytorch/pytorch

WORKDIR ./

COPY ./classification-based/requirements.txt ./
RUN pip3 install -r requirements.txt

COPY .. .

# CMD [ "python", "./classification-based/MAB_main.py" ]