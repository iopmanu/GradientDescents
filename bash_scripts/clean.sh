docker stop tests
docker stop py
docker rm tests
docker rm py
docker rmi gradient_descents-tests:latest
docker rmi python:3.8-alpine
