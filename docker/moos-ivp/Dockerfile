FROM dorowu/ubuntu-desktop-lxde-vnc:xenial

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

ENV HOME /root
ENV MOOSIVP /root/moos-ivp


WORKDIR ${HOME}

ARG DEBIAN_FRONTEND=noninteractive
RUN rm -rf /var/lib/apt/lists/* &&  \
    apt-get update -qq && apt-get install --assume-yes apt-utils > /dev/null &&  \
	apt-get install -y -qq g++ cmake xterm git > /dev/null && \
    apt-get install -y -qq build-essential libeigen3-dev > /dev/null && \
    apt-get install -y -qq vim ibfltk1.3-dev freeglut3-dev libpng-dev libjpeg-dev > /dev/null && \
    apt-get install -y -qq libxft-dev libxinerama-dev libtiff5-dev > /dev/null 
    
# MOOS
RUN git clone https://github.com/ARG-NCTU/moos-ivp-nctu.git moos-ivp

WORKDIR ${MOOSIVP}
RUN ./build-moos.sh && \
    ./build-ivp.sh 

RUN rm -rf /var/lib/apt/lists/* &&  \
    apt-get update -qq && apt-get install --assume-yes apt-utils > /dev/null &&  \
    apt-get install -y -qq byobu  > /dev/null


WORKDIR ${MOOSIVP}

RUN apt-get install -y python-pip python-numpy > /dev/null 

# Opencv 2.4
RUN git clone https://github.com/tonycar12002/opencv-2.4.9-pod.git

RUN apt-get install unzip > /dev/null && \
    cd opencv-2.4.9-pod/pods-svn && make && make install && \
    cd ../opencv-2.4.9-pod && unzip opencv-2.4.9.zip && \
    cp Makefile-odroid Makefile &&\
    make

# MOOS-RobotX
WORKDIR ${MOOSIVP}
RUN git clone https://github.com/tonycar12002/robotx-mit-2014.git 

RUN apt-get update && \
    apt-get install -y -qq libproj-dev > /dev/null && \
    cd robotx-mit-2014 && \
    ./build.sh 

ENV PATH="/root/moos-ivp/bin:${PATH}"
ENV PATH="/root/moos-ivp/robotx-mit-2014/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y -qq network-manager libx11-dev xorg-dev libglu1-mesa-dev freeglut3-dev libglew1.5 libfltk1.3-dev > /dev/null && \
    apt-get install -y -qq net-tools libglew1.5-dev libglu1-mesa libglu1-mesa-dev libgl1-mesa-glx libgl1-mesa-dev > /dev/null && \
    apt-get install -y -qq mesa-utils > /dev/null

EXPOSE 9000

WORKDIR ${MOOSIVP}/ivp/missions/s1_alpha
