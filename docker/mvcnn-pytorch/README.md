Dockerfile for MVCNN-Pytorch (https://github.com/RBirkeland/MVCNN-PyTorch)  

Built image: docker pull peterx7803/mvcnn-pytorch  

# How to run

[#1 container] 1. nvidia-docker run -it --name mvcnn -p 6006:6006 --shm-size 256m -v [modelnet40_views dataset_folder on host]:[Desired path in container] --rm mvcnn bash  

For example:  
nvidia-docker run -it -p 6006:6006 --shm-size 256m -v /home/joinet/MVCNN-PyTorch/modelnet40_views:/tmp/modelnet40_views --rm mvcnn bash  


[#1 container] 2. python controller [dataset_path_in_container/view/classes]  
For example:  
python controller /tmp/modelnet40_views/view/classes  

[#2 container] 3. docker exec -it mvcnn bash  

[#2 container] 4. tensorboard --logdir='logs' --port=6006

[Webviewer] 5. Open webviewer and visit localhost:6006 to view tensorboard

