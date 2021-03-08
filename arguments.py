# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
#
# Modified by jisng.prk@gmail.com

"""argparser configuration"""

import argparse
import os
import torch
import deepspeed


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_model_config_args(parser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--model_select',
                       type=str,
                       default='112m',
                       help='model selection parameter. One of [112m, 112m_half, 345m]')

    return parser


def add_tokenizer_config_args(parser):
    group = parser.add_argument_group('tokenizer', 'tokenizer configuration')

    group.add_argument('--vocab_load_dir',
                       type=str,
                       default='./vocab',
                       help='checkpoint directory name')

    group.add_argument('--vocab_id_dir',
                       type=str,
                       default='vocab_50257',
                       help='checkpoint directory name')

    group.add_argument('--enable_padding',
                       type=str2bool,
                       default=True,
                       help='default: enable padding')

    group.add_argument('--enable_bos',
                       type=str2bool,
                       default=True,
                       help='default: enable bos')

    group.add_argument('--enable_eos',
                       type=str2bool,
                       default=True,
                       help='default: enable eos')

    group.add_argument('--truncated_len',
                       type=int,
                       default=128,
                       help='maximum length of tokenized sentence')

    return parser


def add_fp16_config_args(parser):
    """Mixed precision arguments."""

    group = parser.add_argument_group('fp16', 'fp16 configurations')

    return parser


def add_training_args(parser):
    """Training arguments."""

    group = parser.add_argument_group('train', 'training configurations')

    group.add_argument('--train_mode',
                       type=str,
                       default='pretrain',
                       help='training goal. One of [pretrain, finetune]')

    group.add_argument('--seed',
                       type=int,
                       default=123,
                       help='random seed')

    group.add_argument('--ckpt_dir',
                       type=str,
                       default='./checkpoints',
                       help='directory for save checkpoint')

    group.add_argument('--workspace',
                       type=str,
                       default='test0',
                       help='workspace directory name')

    group.add_argument('--workspace_finetune',
                       type=str,
                       default='test0',
                       help='workspace directory name')

    group.add_argument('--restart',
                       type=bool,
                       default=False,
                       help='restart training')

    group.add_argument('--ckpt_id',
                       type=str,
                       default='epoch:0-step:13000',
                       help='checkpoint directory name')

    group.add_argument('--ckpt_id_finetune',
                       type=str,
                       default='epoch:0-step:13000',
                       help='checkpoint directory name')

    group.add_argument('--train_iters',
                       type=int,
                       default=100_000,
                       help='# of iterations for training')

    group.add_argument('--tr_ratio',
                       type=float,
                       default=0.99,
                       help='ratio of training set in total dataset')

    group.add_argument('--loss_type',
                       type=str,
                       default='lm_loss',
                       help='loss selection argument. Only "lm_loss" is supported')

    parser.add_argument('--wandb_dir',
                        type=str,
                        default='kg_gpt2_0215',
                        help='for setting wandb project')

    group.add_argument('--ckpt_save_steps',
                       type=int,
                       default=2000,
                       help='save checkpoint for every # of steps')

    # distributed training args
    group.add_argument('--distributed-backend',
                       default='nccl',
                       help='which backend to use for distributed '
                       'training. One of [gloo, nccl]')

    group.add_argument('--local_rank',
                       type=int,
                       default=None,
                       help='local rank passed from distributed launcher')

    return parser


def add_evaluation_args(parser):
    """Evaluation arguments."""

    group = parser.add_argument_group('validation', 'validation configurations')

    group.add_argument('--eval_batch_size',
                       type=int,
                       default=128,
                       help='# of batch size for evaluating on each GPU')

    return parser


def add_text_generate_args(parser):
    """Text generate arguments."""

    group = parser.add_argument_group('Text generation', 'configurations')

    group.add_argument('--use_cpu',
                       type=str2bool,
                       default=False,
                       help='use cpu or not. If not, gpu is selected')

    group.add_argument('--gpu_id',
                        type=int,
                        default=0,
                        help='select gpu id')

    group.add_argument('--min_length',
                       type=int,
                       default=100,
                       help='minimum token length')

    group.add_argument('--max_length',
                       type=int,
                       default=120,
                       help='maximum token length')

    group.add_argument('--do_sample',
                       type=bool,
                       default=True,
                       help='generate sequence with sampling')

    group.add_argument('--top_k',
                       type=int,
                       default=30,
                       help='# of k for top k sampling')

    group.add_argument('--temperature',
                       type=float,
                       default=0.9,
                       help='temperature parameter. Lower temperature make the prob distribution sharper')

    group.add_argument('--repetition_penalty',
                       type=float,
                       default=1.2,
                       help='repetition penalty. It is multiplied to temperature')

    group.add_argument('--num_beams',
                       type=int,
                       default=1,
                       help='# of beam search')

    group.add_argument('--port',
                       type=int,
                       default=4000,
                       help='API port')


    return parser


def add_data_args(parser):
    """Train/valid/test data arguments."""

    group = parser.add_argument_group('data', 'data configurations')
    group.add_argument('--config_train',
                       type=str,
                       help='mongoDB configuration for loading training dataset')

    return parser


def add_preprocessing_args(parser):
    group = parser.add_argument_group('preprocessing', 'data preprocessing')

    group.add_argument('--config_src',
                       type=str,
                       help='config file of source mongoDB ')

    group.add_argument('--config_trgt',
                       type=str,
                       help='config file of target mongoDB')

    group.add_argument('--num_process',
                       type=int,
                       default=10,
                       help='number of process used to download the text files from mongoDB')

    group.add_argument('--vocab_size',
                       type=int,
                       default=50257,
                       help='vocab size')

    group.add_argument('--nspecial',
                       type=int,
                       default=100,
                       help='# of special token')

    group.add_argument('--vocab_train_dir',
                       type=str,
                       help='directory saving text files for vocab training')

    group.add_argument('--vocab_train_fname',
                       type=str,
                       help='prefix of text files for vocab training')

    group.add_argument('--vocab_dir',
                       type=str,
                       help='directory for saving trained vocab')

    group.add_argument('--nsample',
                       type=int,
                       help='number of sample used to train vocab')

    group.add_argument('--vocab_prefix',
                       type=str,
                       default='prefix of vocab file')

    return parser


def get_ds_args():
    """Parse all the args."""

    parser = argparse.ArgumentParser(description='PyTorch koGPT2 Model')
    parser = add_model_config_args(parser)
    parser = add_tokenizer_config_args(parser)
    parser = add_training_args(parser)
    parser = add_evaluation_args(parser)
    parser = add_text_generate_args(parser)
    parser = add_data_args(parser)

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    if os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'):
        # We are using (OpenMPI) mpirun for launching distributed data parallel processes
        local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
        local_size = int(os.getenv('OMPI_COMM_WORLD_LOCAL_SIZE'))

        # Possibly running with Slurm
        num_nodes = int(os.getenv('SLURM_JOB_NUM_NODES', '1'))
        nodeid = int(os.getenv('SLURM_NODEID', '0'))

        args.local_rank = local_rank
        args.rank = nodeid*local_size + local_rank
        args.world_size = num_nodes*local_size

    return args


def get_preprocessing_args():
    """Parse all the args."""

    parser = argparse.ArgumentParser(description='PyTorch BERT Model')
    parser = add_preprocessing_args(parser)

    args = parser.parse_args()

    return args


def get_clean_args():
    parser = argparse.ArgumentParser(description='PyTorch BERT Model')
    parser = add_preprocessing_args(parser)
    parser = add_tokenizer_config_args(parser)

    args = parser.parse_args()

    return args