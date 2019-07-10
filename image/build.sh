#!/usr/bin/env bash

VERSION=0.0.1

IMAGE=berkgokden/pri:optimizer-$VERSION

docker build -t $IMAGE .
# docker push $IMAGE

echo $IMAGE
