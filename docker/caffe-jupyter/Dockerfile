FROM bvlc/caffe:gpu

RUN apt-get update -y && apt-get install -y --no-install-recommends \
         git \
         xvfb \
         bzip2 \
         freeglut3-dev && \
     rm -rf /var/lib/apt/lists/*

ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda.sh
RUN sh miniconda.sh -b -p /opt/conda && rm miniconda.sh
ENV PATH /opt/conda/bin:$PATH

WORKDIR /workspace
 
RUN conda install scikit-image jupyter numpy matplotlib pyyaml scipy ipython mkl mkl-include && conda clean -ya

EXPOSE 8888
