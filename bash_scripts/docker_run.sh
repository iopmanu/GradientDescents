cd ..
docker pull python:3.8-alpine
cp main.py docker
cd docker
mkdir source
cp ../source/* source
docker compose build
docker compose up -d
cd ../docker
rm -Rf *.py source
