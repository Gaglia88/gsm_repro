if [ -z "$(docker images -q gsm_comparison 2> /dev/null)" ]; then
  docker build -t gsm_comparison .
fi
docker run -it --gpus all --mount type=bind,source="$(pwd)/../",target=/home/app gsm_comparison