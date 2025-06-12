This is the folder if I don't use local image replace, but build image
directly on Nvidia Ubuntu image. It will cost much more disk space.

**The reason** is if I need to use conda to set the flower env, I need
to import pyproject.toml first, but original code, copy files is the last
step in Dockerfile. The reason is because we want to reuse docker image
layers as many as possible. Because the files path are different(in each
host build folder), even the file contents are same, they will be treated
as different files. In this case, the docker image layer after COPY can
not be reuse. For each container, it will download flower, torch library
again and again. It will consume disk sapce very quickly. And because each
container need to download it again, **it will cost a lot of time!**
This way may works when number of nodes is small, but will doesn't work 
when number of nodes are mass. I tested it doesn't work when # of node =10. 

THUS, we should NOT change the Docker compiler order to put COPY before RUN!

**The correct solution** is to use pre-build custom image to override it. 
This way can maximize the reuse of docker layer image, each container
will share same flower, torch library layer.


In this folder, I keep the old space-waste way, so I can draw the histogram
to show the benefit.

**** TO repreduce ****
1. Docker.py order should be changed, COPY should go before RUN, otherwise,
when I RUN pip install, pyproject.toml doesn't exist yet. I will get ERROR.
comment the previous RUN and uncomment the later RUN.

2. ./space_waste_flower_frame.py 

3. cd output && dcbuild && dcup 

4. After I finish the experiment, change Docker.py back to the correct
version.


|          | 1 server + 2 nodes | 1 server + 10 nodes |
|----------|--------------------|---------------------|
|   Reuse  |        10 G        |         10 G        |
| No Reuse |                    |                     |

