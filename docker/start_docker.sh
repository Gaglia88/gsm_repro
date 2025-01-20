if [ -z "$(docker images -q lucagaglia/gsm_repro 2> /dev/null)" ]; then
  # If uncommented, this will build the image locally
  #docker build -t gsm_repro .
  docker pull lucagaglia/gsm_repro
fi
docker run -it --gpus all -p 8888:8888 --mount type=bind,source="$(pwd)/../",target=/home/app lucagaglia/gsm_repro
