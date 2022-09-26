# MoE inference with Fairseq
This repository contains scripts to streamline the execution of MoE inference using FairSeq

## Install docker

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

### Start docker daemon
```bash
sudo service docker start
```

### Install the NVIDIA Container Toolkit
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Get preprocessed data and MoE pretrained checkpoint
With docker installed, you will want to download the RoBERTa preprocessed data ([see here](https://github.com/facebookresearch/fairseq/blob/moe/examples/roberta/README.pretraining.md#1-preprocess-the-data)) and the 15B MoE model checkpoint ([see here](https://github.com/facebookresearch/fairseq/tree/main/examples/moe_lm)), which you can do by running the following script:

```bash
./get_preprocessed_data.sh
```

This will create a folder named `data` and download/preprocess all data in there.



## Build and run the docker container

You can now build and run the docker container using:

```bash
./build.sh && ./run.sh
```

If you want to only share certain GPUs with the docker container, edit the `run.sh` file to do so.

## Run the MoE inference
Next, head to the `fairseq` folder within the docker container, edit the `world_size` and `distributed-world-size` according to the number of GPUs available on the system, and run the `run_inference.sh` script 

## Save changes to the Docker container
After you exit the Docker container, you can get the container ID of the latest instance by running `docker ps -a`. If you want to log into the same container again, you can do so with:

```bash
docker start <container ID> && docker attach <container ID>
```

If you want to save the changes into a new image, you can do so with:

```bash
docker commit <container ID> new_image_name:tag_name(optional)
```

See here for more info: [StackOverflow link](https://stackoverflow.com/questions/19585028/i-lose-my-data-when-the-container-exits)
