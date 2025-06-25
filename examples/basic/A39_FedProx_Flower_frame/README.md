How to create a base image:
1. copy the template folder `base_template`
```shell-script
$ cp -r base_template flower_base
```
2. copy the flower task folder into new folder
```shell-script
$ cp -r quickstart-pytorch flower_base/
```
3. use md5 create the staged name of task folder
```shell-script
$ echo -n "/quickstart-pytorch" | md5sum
53be52b6016ab1f879517691a07f59ac  -
$ mv quickstart-pytorch 53be52b6016ab1f879517691a07f59ac
```
4. replace the name placehoder in Dockerfile

Or use bash script to generate
```
bash generate_base_image.sh flower_test quickstart-pytorch /quickstart-pytorch
```

After that, replace image name in flower_frame.py
```
imageName = 'fedprox_base'
dirName = './fedprox_base'
```
