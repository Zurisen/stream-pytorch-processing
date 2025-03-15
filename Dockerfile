FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch==2.6.0 torchvision==0.21.0
# Copy the rest of the application code
COPY src/ ./src/
COPY models/ ./models/
COPY etc/ ./etc/
COPY result/ ./result/

# Set the command to run the application
CMD ["python", "src/main.py"]
