FROM duckietown/rpi-duckiebot-base

WORKDIR /home/software
RUN git clone -b ncsdk2 https://github.com/movidius/ncsdk.git && cd ncsdk && sudo make install
WORKDIR /home/software
