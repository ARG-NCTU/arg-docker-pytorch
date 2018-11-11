# moos-ivp-nctu-docker
This docker file contains two main repos
* MOOs-IvP - Fork the repo from [MIT MOOS website](http://oceanai.mit.edu/moos-ivp/pmwiki/pmwiki.php?n=Main.HomePage) in January 2018
* RobotX-2014-MIT - The first place code in the 2014 RobotX competition from [MIT team](http://oceanai.mit.edu/robotx/pmwiki/pmwiki.php?n=Main.HomePage)

## How to get docker images
There are 2 ways to get docker images

1. build from dockfer file
```
$ docker build -t moos-ivp-nctu -f Dockerfile .
```
2. get docker from docker hub
```
$ docker pull tonycar12002/moos-ivp-nctu
```

## How to run
```
$ xhost +local:root
$ docker run -it --rm -p 8080:80 moos-ivp-nctu
or 
$ docker run -it --rm -p 8080:80 tonycar12002/moos-ivp-nctu
Open a browser and type docker_ip:port_num, ex: http://172.17.0.1:8080/
```

## Test the moos-ivp 
1. Test simple mission
Open a Terminal
```
container# cd /root/moos-ivp/ivp/missions/s1_alpha
container# ./launch.sh
```
you will see MOOSApp pMarinneViwer pop out and then press button "deploy"


![pMarineviewer](https://github.com/tonycar12002/moos-ivp-nctu-docker/blob/master/pMarineviewer.png)

</br>

2. Test RobotX 2014 Task1-Qualifying
```
container# cd /root/moos-ivp/robotx-mit-2014/ATTIC/missions_test/Task1
container# ./launch.sh
```
press button "deploy" </br>
![RobotX2014Task1](https://github.com/tonycar12002/moos-ivp-nctu-docker/blob/master/Task1.png)


## Still Working On
This is the beta version of our docker file. We use VNC to be the GUI interface for opening MOOS-IvP. We are still working on using MOOS without VNC, and just use terminal.

## Contacts
If you have any question please contact to us.
* Tony Hsiao <tonycar12002@gmail.com>
