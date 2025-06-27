# Use the official JupyterLab image as base
FROM jupyter/scipy-notebook:latest

# Install additional system tools if needed 
USER root
RUN apt-get update && apt-get install -y wget
USER jovyan

# Install required Python packages
RUN pip install numpy matplotlib scikit-image opencv-python

# Copy code into the container
COPY Z_Scope_Processing /home/jovyan/work/Z_Scope_Processing

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZEkRrKTXIPz3WpLIEZDwAohS9X41uX4W' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZEkRrKTXIPz3WpLIEZDwAohS9X41uX4W" -O /home/jovyan/work/demo.tiff && rm -rf /tmp/cookies.txt


# Set the working directory
WORKDIR /home/jovyan/work
