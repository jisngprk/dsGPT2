# dsGPT2: DeepSpeedGPT2

## Introduction
The goal is as follows:
* Build project that trains GPT2 with DeepSpeed library
* Check that the limitations of training the large scaled language model in limited resource
* Develop conversational model with GPT2 pretrained with conversational data

## Try It out

#### Pretrained Model API 
###### checkpoint: 142k step 
 
```
GET http://34.82.253.174:4000/generate?sentence=삼성전자와 테슬라는 협업을 
```

![1](https://user-images.githubusercontent.com/24973802/111060333-76c30d80-84df-11eb-800b-0bdb5495330e.PNG)

![2](https://user-images.githubusercontent.com/24973802/111060353-aa059c80-84df-11eb-9168-683c9f300b9e.PNG)

#### Conversational Model API 
###### checkpoint: 78k step


```
GET http://34.82.253.174:4000/generate?sentence=점심 뭐 먹을래? 
```

![chat-1](https://user-images.githubusercontent.com/24973802/111060368-c0135d00-84df-11eb-8a7a-ed13a1334373.PNG)

![chat-2](https://user-images.githubusercontent.com/24973802/111060403-ef29ce80-84df-11eb-9c89-52648fb81b1f.PNG)

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
        - Lazy loading   
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
0. Make directory './vocab', './checkpoints' and make './config/db_config.json'. Also make './data_files' directory to save vocab training files.

1. You can build the image with Dockerfile-dev. It download deepspeed image used for torch 1.5^M
    ```shell script
    docker build -t IMAGE_TAG -f Dockerfile-dev .
    ```


2. Run .sh files with the image using the commands:^M
    ```shell script
    docker run -d --name CONTAINER_NAME -e WANDB_API_KEY=WANDB_KEY --gpus='"device=0,1"' --network host -v PROJECT_DIR:/usr/src/app -w /usr/src/app DOCKER_IMAGE bash scripts/ds_trainer.sh
    ```

Maybe you can change some gpus setting in the '--gpus' option or in scripts dpending on your node environment
* WANDB_API_KEY: wandb api key that you can get in your wandb account
* PROJECT_DIR: directory in which you download this project

The deepspeed or torch version can have dependency with the gpu drivers, torch versions etc. You maybe check your environment.


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
bash scripts/vocab_builder.sh
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
        
    usage: ds_trainer.py [-h] [--model_select MODEL_SELECT]
                         [--vocab_load_dir VOCAB_LOAD_DIR]
                         [--vocab_id_dir VOCAB_ID_DIR]
                         [--enable_padding ENABLE_PADDING]
                         [--enable_bos ENABLE_BOS] [--enable_eos ENABLE_EOS]
                         [--truncated_len TRUNCATED_LEN] [--train_mode TRAIN_MODE]
                         [--seed SEED] [--ckpt_dir CKPT_DIR]
                         [--workspace WORKSPACE]
                         [--workspace_finetune WORKSPACE_FINETUNE]
                         [--restart RESTART] [--ckpt_id CKPT_ID]
                         [--ckpt_id_finetune CKPT_ID_FINETUNE]
                         [--train_iters TRAIN_ITERS] [--tr_ratio TR_RATIO]
                         [--loss_type LOSS_TYPE] [--wandb_dir WANDB_DIR]
                         [--ckpt_save_steps CKPT_SAVE_STEPS]
                         [--distributed-backend DISTRIBUTED_BACKEND]
                         [--local_rank LOCAL_RANK]
                         [--eval_batch_size EVAL_BATCH_SIZE] [--use_cpu USE_CPU]
                         [--gpu_id GPU_ID] [--min_length MIN_LENGTH]
                         [--max_length MAX_LENGTH] [--do_sample DO_SAMPLE]
                         [--top_k TOP_K] [--temperature TEMPERATURE]
                         [--repetition_penalty REPETITION_PENALTY]
                         [--num_beams NUM_BEAMS] [--port PORT]
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
    
    tokenizer:
      tokenizer configuration
    
      --vocab_load_dir VOCAB_LOAD_DIR
                            checkpoint directory name
      --vocab_id_dir VOCAB_ID_DIR
                            checkpoint directory name
      --enable_padding ENABLE_PADDING
                            default: enable padding
      --enable_bos ENABLE_BOS
                            default: enable bos
      --enable_eos ENABLE_EOS
                            default: enable eos
      --truncated_len TRUNCATED_LEN
                            maximum length of tokenized sentence
    
    train:
      training configurations
    
      --train_mode TRAIN_MODE
                            training goal. One of [pretrain, finetune]
      --seed SEED           random seed
      --ckpt_dir CKPT_DIR   directory for save checkpoint
      --workspace WORKSPACE
                            workspace directory name
      --workspace_finetune WORKSPACE_FINETUNE
                            workspace directory name
      --restart RESTART     restart training
      --ckpt_id CKPT_ID     checkpoint directory name
      --ckpt_id_finetune CKPT_ID_FINETUNE
                            checkpoint directory name
      --train_iters TRAIN_ITERS
                            # of iterations for training
      --tr_ratio TR_RATIO   ratio of training set in total dataset
      --loss_type LOSS_TYPE
                            loss selection argument. Only "lm_loss" is supported
      --ckpt_save_steps CKPT_SAVE_STEPS
                            save checkpoint for every # of steps
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
    
      --use_cpu USE_CPU     use cpu or not. If not, gpu is selected
      --gpu_id GPU_ID       select gpu id
      --min_length MIN_LENGTH
                            minimum token length
      --max_length MAX_LENGTH
                            maximum token length
      --do_sample DO_SAMPLE
                            generate sequence with sampling
      --top_k TOP_K         # of k for top k sampling
      --temperature TEMPERATURE
                            temperature parameter. Lower temperature make the prob
                            distribution sharper
      --repetition_penalty REPETITION_PENALTY
                            repetition penalty. It is multiplied to temperature
      --num_beams NUM_BEAMS
                            # of beam search
      --port PORT           API port
    
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
* Data source: 국립국어원 모두의 말뭉치 ver 1.0 
   - 웹 말뭉치, 신문 말뭉치, 문어 말뭉치, 구어 말뭉치, 메신저 말뭉치


### Loss - pretrain 

| # of parameters  | Step | Loss | PPL
|---|---|---|---|
|112M| 142k| ~ 3.9 | ~ 48.95  

![pretrain-loss](https://user-images.githubusercontent.com/24973802/111060533-e259aa80-84e0-11eb-891d-ec12e9a475c8.png)


### Loss - finetune

| # of parameters | Step |  Loss | Acc   
|---|---|---|---|
|112M|  78k | ~ 0.048 | ~ 0.985 

![finetune-loss](https://user-images.githubusercontent.com/24973802/111060658-d7534a00-84e1-11eb-9c60-0b3e9b933c30.png)

![finetune-acc](https://user-images.githubusercontent.com/24973802/111060657-d4f0f000-84e1-11eb-813a-69377c83e97c.png) 

