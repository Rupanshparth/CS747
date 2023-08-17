#!/bin/bash
echo "Running Docker Container"
sudo docker run \
  -it \
  -v $(pwd):$(pwd) -w $(pwd)\
  docker.cse.iitb.ac.in/cs747
