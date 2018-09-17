FROM pytorch/pytorch

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    xvfb \
    freeglut3-dev && \
    rm -rf /var/lib/apt/lists/*
 
RUN conda install jupyter numpy pyyaml scipy ipython mkl mkl-include && conda clean -ya

