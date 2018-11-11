This is based on the Dockerfile for peterx's base_image


### Build 

```
$ docker build -t argnctu/ros-caffe-ssd -f Dockerfile .
```
Or replace argnctu to your dockerhub account

### Train

If want to train SSD on workstation, need to modify examples/ssd/ssd_pascal.py:
Line 332 gpus = "0, 1, 2, 3" to "0"
Line 337 batch_size = 32 to batch_size = 16
Line 338 accum_batch_size = 32 to accum_batch_size = 16
