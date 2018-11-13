FROM allenou/ncs_base
#FROM andyser/ncs_mser_textspotting:no_sh_file
WORKDIR /home/
COPY street_en_harvest.graph .
RUN git clone https://github.com/AndySer37/duckietown_text
RUN /bin/bash -c "cd /home/duckietown_text/ && source /opt/ros/kinetic/setup.bash && catkin_make -C catkin_ws/"
RUN pip install Pillow

WORKDIR /home/duckietown_text/
COPY run_ncs_mser.sh .

CMD [ "./run_ncs_mser.sh" ]
