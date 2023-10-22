FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Update system packages and install dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx build-essential

# Copy the necessary files and directories to the container
COPY sort sort/
COPY . .

# Set up conda environment and install dependencies
RUN conda create --prefix ./env python==3.8 -y
RUN /bin/bash -c "source activate ./env && pip install numpy==1.24.3"
RUN /bin/bash -c "source activate ./env && pip install -r requirements.txt"

# Command to run when the container starts
CMD [ "/bin/bash", "-c", "source activate ./env && python main.py && video_to_frame.py && client.py && python add_missing_data.py && python visualize.py"]
