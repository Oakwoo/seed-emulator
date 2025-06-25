#!/bin/bash

if [ $# -ne 3 ]; then
  echo "Usage: $0 <IMAGE_NAME> <HOST_PATH> <TARGET_PATH>"
  exit 1
fi

image_name=$1
host_path=$2
target_path=$3

# copy the template folder
cp -r base_template ${image_name}

# copy the flower task folder into new folder
staged_path=$(echo -n "${target_path}" | md5sum | cut -d ' ' -f1)
cp -r ${host_path} ${image_name}/${staged_path}

# replace the placehoder in Dockerfile
# it is better to use | instead of | to escaping, becuase target path has / will mess up
sed -i "s|TARGET_PATH|${target_path}|g" ${image_name}/Dockerfile
sed -i "s|STAGED_PATH|${staged_path}|g" ${image_name}/Dockerfile

