FROM quay.io/mondrianscwgs/miniconda3:23.3.1-0
ARG VERSION
RUN apt-get update -y && apt install build-essential -y && rm -rf /var/lib/apt/lists/*
RUN pip install numpy cython
RUN pip install git+https://github.com/mondrian-scwgs/scgenome.git@$VERSION
