if [ -z "$(docker images -q lucagaglia/gsm_comparison 2> /dev/null)" ]; then
  # If uncommented, this will build the image locally
  #docker build -t gsm_comparison .
  docker pull lucagaglia/gsm_comparison
fi
docker run -it --gpus all --mount type=bind,source="$(pwd)/../",target=/home/app lucagaglia/gsm_comparison
