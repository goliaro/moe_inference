# MoE inference with Fairseq
This repository contains scripts to streamline the execution of MoE inference using FairSeq

## Install docker
First, you will need to install Docker and the NVIDIA Container Toolkit, which enables Docker containers to use the host machine's GPU(s).

<details>
<summary>Toggle here for instructions</summary>
<br>

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

</details>

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

## Known issues and troubleshooting

<details>
<summary>Wrong MODEL_PATH definition causes inference script to hang/error</summary>
<br>

After having pre-processed the data and downloaded the checkpoint, you will have a folder called `data/en_moe_lm_15b`, which is mounted at `/mnt/data/en_moe_lm_15b` in the Docker container (if you used the provided scripts without making any edits). The folder contains a file named `model-shared.pt`, and several `model-rank-<N>.pt` files. Notice that there is no `model.pt` file. Despite this, you will need to pass a `--path` argument to `fairseq_cli.eval_lm` that points to the non-existing `model.pt` file. For instance, in the [scripts/run_inference.sh](./scripts/run_inference.sh) script, we pass `--path $MODEL_PATH`, after having set `MODEL_PATH=/mnt/en_moe_lm_15b/model.pt`. This might seem odd, but any other choice for the argument will result either in error, or will make the `eval_lm` CLI hang, as follows:

**If you set `MODEL_PATH=/mnt/en_moe_lm_15b/model-shared.pt`** you will get an error similar to the one below:

```bash
2022-09-28 22:43:29 | INFO | fairseq.checkpoint_utils | load_model_ensemble_and_task is_moe=True
Traceback (most recent call last):
  File "/opt/conda/lib/python3.7/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/opt/conda/lib/python3.7/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/workspace/fairseq/fairseq_cli/eval_lm.py", line 448, in <module>
    cli_main()
  File "/workspace/fairseq/fairseq_cli/eval_lm.py", line 444, in cli_main
    distributed_utils.call_main(convert_namespace_to_omegaconf(args), main)
  File "/workspace/fairseq/fairseq/distributed/utils.py", line 358, in call_main
    join=True,
  File "/opt/conda/lib/python3.7/site-packages/torch/multiprocessing/spawn.py", line 230, in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
  File "/opt/conda/lib/python3.7/site-packages/torch/multiprocessing/spawn.py", line 188, in start_processes
    while not context.join():
  File "/opt/conda/lib/python3.7/site-packages/torch/multiprocessing/spawn.py", line 150, in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
torch.multiprocessing.spawn.ProcessRaisedException: 

-- Process 0 terminated with the following error:
Traceback (most recent call last):
  File "/opt/conda/lib/python3.7/site-packages/torch/multiprocessing/spawn.py", line 59, in _wrap
    fn(i, *args)
  File "/workspace/fairseq/fairseq/distributed/utils.py", line 335, in distributed_main
    main(cfg, **kwargs)
  File "/workspace/fairseq/fairseq_cli/eval_lm.py", line 391, in main
    is_moe=is_moe or is_base_moe,
  File "/workspace/fairseq/fairseq/checkpoint_utils.py", line 450, in load_model_ensemble_and_task
    raise IOError("Model file not found: {}".format(filename))
OSError: Model file not found: /mnt/en_moe_lm_15b/model-shared-rank-0.pt
```

**If you set `MODEL_PATH=/mnt/en_moe_lm_15b/model.pt` and rename `/mnt/en_moe_lm_15b/model-shared.pt` to `/mnt/en_moe_lm_15b/model.pt` you will get the fairseq script to hang as described below:**

The Fairseq script (called by `run_inference.sh`) will run for 1min or so, with several threads reaching high CPU utilization (~100%). You will also see the RAM usage go up progressively, together with the GPU memory usage. At some point, after around ~60-70 GB of RAM have been allocated, all the threads will drop to 0% CPU utilization, and the application will idle indefinitely. GPU memory usage will remain high and constant. No additional information will be printed to stdout or stderr.

Stdout tail:
```bash
2022-09-26 16:09:44 | INFO | fairseq.tasks.language_modeling | dictionary: 50264 types
2022-09-26 16:09:44 | INFO | fairseq.checkpoint_utils | load_model_ensemble_and_task is_moe=True
2022-09-26 16:09:56 | INFO | torch.distributed.distributed_c10d | Added key: store_based_barrier_key:3 to store for rank: 0
2022-09-26 16:09:56 | INFO | torch.distributed.distributed_c10d | Rank 0: Completed store-based barrier for key:store_based_barrier_key:3 with 4 nodes.
2022-09-26 16:09:56 | INFO | torch.distributed.distributed_c10d | Added key: store_based_barrier_key:4 to store for rank: 0
2022-09-26 16:09:56 | INFO | torch.distributed.distributed_c10d | Rank 0: Completed store-based barrier for key:store_based_barrier_key:4 with 4 nodes.
2022-09-26 16:09:56 | INFO | torch.distributed.distributed_c10d | Added key: store_based_barrier_key:5 to store for rank: 0
2022-09-26 16:09:56 | INFO | torch.distributed.distributed_c10d | Rank 0: Completed store-based barrier for key:store_based_barrier_key:5 with 4 nodes.
2022-09-26 16:09:56 | INFO | torch.distributed.distributed_c10d | Added key: store_based_barrier_key:6 to store for rank: 0
2022-09-26 16:09:56 | INFO | torch.distributed.distributed_c10d | Rank 0: Completed store-based barrier for key:store_based_barrier_key:6 with 4 nodes.
2022-09-26 16:09:56 | INFO | torch.distributed.distributed_c10d | Added key: store_based_barrier_key:7 to store for rank: 0
2022-09-26 16:09:56 | INFO | torch.distributed.distributed_c10d | Rank 0: Completed store-based barrier for key:store_based_barrier_key:7 with 4 nodes. 
```

Nvidia-smi:
```bash
gabrieleoliaro@ip-172-31-6-160:~/moe_inference$ nvidia-smi
Tue Sep 27 20:26:34 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  Off  | 00000000:00:1B.0 Off |                    0 |
| N/A   37C    P0    50W / 300W |   1392MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-SXM2...  Off  | 00000000:00:1C.0 Off |                    0 |
| N/A   35C    P0    51W / 300W |   1390MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-SXM2...  Off  | 00000000:00:1D.0 Off |                    0 |
| N/A   40C    P0    54W / 300W |   1464MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-SXM2...  Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   38C    P0    53W / 300W |   1464MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A   2870431      C   /opt/conda/bin/python            1389MiB |
|    1   N/A  N/A   2870432      C   /opt/conda/bin/python            1387MiB |
|    2   N/A  N/A   2870433      C   /opt/conda/bin/python            1461MiB |
|    3   N/A  N/A   2870434      C   /opt/conda/bin/python            1461MiB |
+-----------------------------------------------------------------------------+
```

If you interrupt the process with Ctrl+C, you will get the following Stderr log, indicating that the N worker threads (where N is determined by the world size parameter we set in `run_inference.sh`) are stuck while loading the state dict from the pretrained model (`model.load_state_dict` function). Below, the full stderr and a screenshot for the N=4 worker threads case.

[error.log](https://github.com/gabrieleoliaro/fairseq_exp/files/9665756/error.log)

![runtime_error](https://user-images.githubusercontent.com/6480808/192799673-e310cfc1-337e-4fda-8abb-98759ade420a.png)


</details>
