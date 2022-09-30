from tensorflow/tensorflow:2.7.2-gpu
RUN apt update -y
RUN apt install vim -y
WORKDIR /causal
# Later: Mount src and copy other directories individually.
COPY dat dat
COPY pre-trained pre-trained
COPY requirements.txt .
# COPY . .
RUN pip install -r requirements.txt
