FROM ubuntu:16.04
#installing ubuntu essentials
RUN apt-get update
RUN apt-get -y install python3-setuptools python3-dev python3-pip build-essential
RUN apt-get install -y python3-tk

# create directory
RUN mkdir /workdirectory

WORKDIR /workdirectory

# Copy over files
COPY . ./

RUN pip3 install -r requirements.txt

CMD [ "python3", "train.py"]
