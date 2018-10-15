FROM pytorch/pytorch

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    xvfb \
    freeglut3-dev && \
    rm -rf /var/lib/apt/lists/*
 
RUN conda install jupyter numpy matplotlib pyyaml scipy ipython mkl mkl-include pandas && conda clean -ya

RUN conda install scikit-image && conda clean -ya
