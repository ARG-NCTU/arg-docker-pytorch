

### To Build
```
$ docker build -t pygame -f Dockerfile . 
```

### To Run
```
$ docker run -dt -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix pygame
```
