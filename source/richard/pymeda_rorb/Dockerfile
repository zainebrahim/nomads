FROM ubuntu:16.04
#installing ubuntu essentials
RUN apt-get update
RUN apt-get -y install python3 python-dev python3-dev \
     build-essential python-pip libnuma-dev
RUN apt-get install -y python3-tk

# create directory
RUN mkdir /workdirectory

WORKDIR /workdirectory

# Copy over files
COPY *.py ./
COPY *.cfg ./
COPY requirements.txt ./

RUN pip3 install -r requirements.txt

CMD [ "echo", "Hello"]
