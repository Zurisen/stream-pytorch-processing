# stream-pytorch-processing
Necessary scripts for processing a RTMP streaming of images through a Pytorch neural network

# To build for linux
docker buildx build --platform linux/amd64,linux/arm64 -t cmto/video-processing-linux --push .