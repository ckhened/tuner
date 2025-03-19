#!/bin/bash

docker build . -t dq:ch --build-arg HTTP_PROXY=$HTTP_PROXY --build-arg HTTPS_PROXY=$HTTPS_PROXY 
