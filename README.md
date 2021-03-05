# dsGPT2: DeepSpeedGPT2

## Introduction
The goal is as follows:
* Build project that trains GPT2 with DeepSpeed library
* Identify that the limitations of training the large scaled language model in limited resource
* Develop conversational model with GPT2 pretrained with conversation

## Dependencies
* MongoDB
* DeepSpeed
* PyTorch
* Huggingface / transformers  
* Huggingface / tokenizers

## Key features
* MongoWrapper
    * *Fast* 
        - Fetching document with indexing collection (0.1ms/1doc)
    * *Memory-Efficient* 
        - Lazy loading (memory usage TODO)  
    * *Seamless integration* 
        - Collections accessible by unified index      
    
 * Pipeline
    * Manage with environment with dockers
    * Run each pipeline with bash file
    * Easy to use   
    
 * Deepspeed engine
    * Train large model with limited resource
    * Try to use it: https://www.deepspeed.ai/

 * CrossEntropy loss with large batch size
    * FP16 has range of +-65,504
    * To avoid the overflow, the mean of sample mean over token loss is used instead of global average of each token loss
    * The mean of sample mean can estimate the population mean 
    
## Usage

### Note
* The project use a predefined schema with mongoDB
* You should follow the schema to apply all the pipeline of this project

### Document Schema

Data collection should be indexed by the 'idx' field to get fast access.

```json
{
   "_id": "ObjectID",
   "idx": 0,
   "form": "original text",
   "filt_text": "filtered text"
}
```
* "form": Original text field
* "filt_text": filtered text field

Also, the DB should have 'meta_info' collection. The collection has the schema as follows:
```json
{
   "_id": "ObjectID",
   "collection_name": "collection name",
   "num_docs": 110000
}
```

### Config Schema

MongoWrapper requires a config file that describes which collections should be connected
The files are located in config directory

```json

{
  "MONGO_CONNECTION_STRING":"MongoDB connection string",
  "MONGO_CONNECTION_DB":"Collection Name",
  "COLLECTIONS": ["collection name"]
}
```
* "COLLECTIONS": list all the collection names to integrate in a single index list

### Build/Run Docker

1. Download the deepspeed image from hub 
    - I have used torch 1.5 with cuda 10.1
        - docker pull deepspeed/deepspeed:v031_torch15_cuda101
       
2. Install required packages in a container from the image
    ```shell script
    pip install requirements.txt
    ```

3. Commit the container as image

3. Run .sh files with the image using the commands:
```shell script
docker run -d --name CONTAINER_NAME -e WANDB_API_KEY=WANDB_KEY --gpus='"device=0,1"' --network host -v PROJECT_DIR:/usr/src/app -w /usr/src/app deepspeed/deepspeed:v031_torch15_cuda101 bash scripts/ds_trainer.sh
```
Maybe you can change some gpus setting in the '--gpus' option or in scripts dpending on your node environment
* WANDB_API_KEY: wandb api key that you can get in your wandb account
* PROJECT_DIR: directory in which you download this project


### Download the vocab training files
The script run vocab_downloader.py <br>
It downloads data used for training vocab from collections to separated text files with multiprocessing.
It consumes about 22min to fetch 50M text lines with 30 number of processes.
Your data should be prepared in MongoDB with specified form

```shell script
bash scripts/vocab_downloader.sh
```

### Train the vocab
The script run vocab_trainer.py <br>
It trains ByteLevelBPETokenizer.

```shell script
bash scripts/vocab_trainer.sh
```
### Train the model
The script run ds_trainer.py
It trains GPT2LMHeadModel with data collections specified in db_config.json
Also, It uses ds_config.json which handles the behavior of DeepSpeed engine
such as optimizer, lr scheduler

```shell script
bash scripts/ds_trainer.sh
```

The detail of command-line usage is as follows:

    
    usage: ds_trainer.py [-h] [--model_select MODEL_SELECT] [--seed SEED]
                         [--ckpt_dir CKPT_DIR] [--workspace WORKSPACE]
                         [--restart RESTART] [--ckpt_id CKPT_ID]
                         [--vocab_load_dir VOCAB_LOAD_DIR]
                         [--vocab_id_dir VOCAB_ID_DIR] [--train_iters TRAIN_ITERS]
                         [--tr_ratio TR_RATIO] [--loss_type LOSS_TYPE]
                         [--wandb_dir WANDB_DIR]
                         [--distributed-backend DISTRIBUTED_BACKEND]
                         [--local_rank LOCAL_RANK]
                         [--eval_batch_size EVAL_BATCH_SIZE]
                         [--config_train CONFIG_TRAIN] [--deepspeed]
                         [--deepspeed_config DEEPSPEED_CONFIG] [--deepscale]
                         [--deepscale_config DEEPSCALE_CONFIG] [--deepspeed_mpi]
    
    PyTorch koGPT2 Model
    
    optional arguments:
      -h, --help            show this help message and exit
      --wandb_dir WANDB_DIR
                            for setting wandb project
    
    model:
      model configuration
    
      --model_select MODEL_SELECT
                            model selection parameter. One of [112m, 112m_half,
                            345m]
    
    train:
      training configurations
    
      --seed SEED           random seed
      --ckpt_dir CKPT_DIR   directory for save checkpoint
      --workspace WORKSPACE
                            workspace directory name
      --restart RESTART     restart training
      --ckpt_id CKPT_ID     checkpoint directory name
      --vocab_load_dir VOCAB_LOAD_DIR
                            checkpoint directory name
      --vocab_id_dir VOCAB_ID_DIR
                            checkpoint directory name
      --train_iters TRAIN_ITERS
                            # of iterations for training
      --tr_ratio TR_RATIO   ratio of training set in total dataset
      --loss_type LOSS_TYPE
                            loss selection argument. Only "lm_loss" is supported
      --distributed-backend DISTRIBUTED_BACKEND
                            which backend to use for distributed training. One of
                            [gloo, nccl]
      --local_rank LOCAL_RANK
                            local rank passed from distributed launcher
    
    validation:
      validation configurations
    
      --eval_batch_size EVAL_BATCH_SIZE
                            # of batch size for evaluating on each GPU
    
    Text generation:
      configurations
    
    data:
      data configurations
    
      --config_train CONFIG_TRAIN
                            mongoDB configuration for loading training dataset
    
    DeepSpeed:
      DeepSpeed configurations
    
      --deepspeed           Enable DeepSpeed (helper flag for user code, no impact
                            on DeepSpeed backend)
      --deepspeed_config DEEPSPEED_CONFIG
                            DeepSpeed json configuration file.
      --deepscale           Deprecated enable DeepSpeed (helper flag for user
                            code, no impact on DeepSpeed backend)
      --deepscale_config DEEPSCALE_CONFIG
                            Deprecated DeepSpeed json configuration file.
      --deepspeed_mpi       Run via MPI, this will attempt to discover the
                            necessary variables to initialize torch distributed
                            from the MPI environment


### Data
| Data  |  # of Documents  |
|---|---|
|Newspaper|  37.2M| 
|Spoken|  20.6M | 
|Web|  10.5M | 
|Written|  27.2M
|------|  ------ 
|Total|  95.5M 

* Word count ~= 2B
* 모두의 말뭉치 사용


### Evaluate

### Result