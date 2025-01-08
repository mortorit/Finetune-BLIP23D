FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG USERNAME
ARG USER_UID
ARG USER_GID=$USER_UID
ARG USER_GNAMES
ARG USER_GADD_ARGS
ARG PIP_REQ_FILE
ARG APT_REQ_FILE

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y && apt update
RUN apt-get install -y \
        build-essential ca-certificates python3.9 python3.9-dev python3.9-distutils git vim wget cmake python3-pip
RUN ln -sv /usr/bin/python3.9 /usr/bin/python
RUN ln -svf /usr/bin/python3.9 /usr/bin/python3

RUN apt install python3-distutils

# Add user with its groups
USER 0
RUN IFS=";" read -a myarr <<< "$USER_GADD_ARGS" && \
    unset 'myarr[${#arr[@]}-1]' && \
    for i in ${!myarr[@]}; do \
        if [[ ${myarr[$i]} != "sudo"* ]]; then \
            groupadd -f -g ${myarr[$i]}; \
        fi \
    done

RUN  useradd -u $USER_UID -g $USER_GID -m $USERNAME

RUN for g in $USER_GNAMES; do \
        usermod -a -G $g $USERNAME; \
    done
# Allow sudo
RUN     apt-get install -y sudo \
        && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
        && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME

ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py

# -- Create folders under root for following services
USER 0

# Enable jupyter
RUN mkdir -p /.local
RUN chmod -R 777 /.local

# Enable avalanche
RUN mkdir -p /.avalanche
RUN chmod -R 777 /.avalanche

# Enable wandb
RUN mkdir -p /.config
RUN chmod -R 777 /.config
RUN touch /.netrc
RUN chmod 777 /.netrc
RUN mkdir -p /.cache
RUN chmod -R 777 /.cache

USER $USERNAME

# -- Create folders under root for following services

# install dependencies
USER 0
#ADD apt_requirements.txt /apt_requirements.txt
ADD $APT_REQ_FILE /apt_requirements.txt
RUN apt-get update && cat /apt_requirements.txt | xargs apt-get install -y 
    #apt-get install -y ffmpeg libsm6 libxext6
 

WORKDIR ~/exp

RUN git clone https://github.com/baaivision/Uni3D.git

ENV PATH=/opt/conda/bin:$PATH

# Download and install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Verify the installation
RUN conda --version

RUN cd Uni3D && source /opt/conda/bin/activate && conda create -n uni3d python=3.8 && conda activate uni3d && conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -y -c pytorch -c nvidia && pip install -r requirements.txt && pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
