# Build image from peterx7803/pointnet which included Tensorflow

# Built image: docker pull peterx7803/mvcnn-pytorch

FROM peterx7803/pointnet:latest


# Install Pytorch

# ==================================================================
# tools
# ------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    wget \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/* 

# ==================================================================
# python
# ------------------------------------------------------------------

RUN python -m pip --no-cache-dir install --upgrade \
        setuptools \
        pip 

RUN python -m pip --no-cache-dir install --upgrade \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-learn==0.20.0 \
        matplotlib \
        Cython 


# ==================================================================
# pytorch
# ------------------------------------------------------------------

RUN python -m pip --no-cache-dir install --upgrade \
    	numpy \
    	torchvision_nightly 

RUN python -m pip --no-cache-dir install --upgrade \
        torch_nightly -f \
        https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html


# ldconfig




WORKDIR /workspace


RUN git clone https://github.com/RBirkeland/MVCNN-PyTorch

RUN pip install jupyter

WORKDIR /workspace/MVCNN-PyTorch

RUN mkdir checkpoint && mkdir logs

