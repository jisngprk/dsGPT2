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
* Huggingface - transformers (GPT2LMHeadModel)
* Huggingface - tokenizers (ByteLevelBPETokenizer)

## Key features
* MongoWrapper
    * *Fast* - fetching document with indexing collection (0.1ms/1doc)
    * *Memory-Efficient* - lazy loading (memory usage TODO)  
    * *Seamless integration* - collections accessible by unified index      
    
 * Pipeline
    * Manage with environment with dockers
    * Run each pipeline with bash file    
    
 * Deepspeed engine
    * Train large model with limited resource
    * Try to use it: https://www.deepspeed.ai/

## Usage

### Note
* The project use a predefined schema with mongoDB
* You should follow the schema to apply all the pipeline of this project

### Document Schema

Collection should be indexed by the 'idx' field to get fast access.

```json
{
   "_id": ObjectID,
   "idx": 0,
   "form": "original text",
   "filt_text": "filtered text"
}
```
* "form": Original text field
* "filt_text": filtered text field

### Config Schema


```json

{
  "MONGO_CONNECTION_STRING":"MongoDB connection string",
  "MONGO_CONNECTION_DB":"Collection Name",
  "COLLECTIONS": ["collection name"]
}
```
* "COLLECTIONS": list all the collection names to integrate in a single index list
### Build/Run Docker

### Download the vocab training files

### Train the vocab

### Train the vodel

The detail of command-line usage is as follows:

    usage: ds_trainer.py [-h] [--model_select MODEL_SELECT] [--seed SEED]
                         [--ckpt_dir CKPT_DIR] [--workspace WORKSPACE]
                         [--train_iters TRAIN_ITERS] [--tr_ratio TR_RATIO]
                         [--loss_type LOSS_TYPE] [--wandb_dir WANDB_DIR]
                         [--distributed-backend DISTRIBUTED_BACKEND]
                         [--local_rank LOCAL_RANK]
                         [--eval_batch_size EVAL_BATCH_SIZE] [--load_dir LOAD_DIR]
                         [--ckpt_id CKPT_ID] [--config_train CONFIG_TRAIN]
                         [--deepspeed] [--deepspeed_config DEEPSPEED_CONFIG]
                         [--deepscale] [--deepscale_config DEEPSCALE_CONFIG]
                         [--deepspeed_mpi]
    
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
      --load_dir LOAD_DIR   checkpoint parent directory
      --ckpt_id CKPT_ID     checkpoint id directory
    
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


### Evaluate

### Result