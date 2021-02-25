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

"""argparser configuration"""

import argparse
import os
import torch
import deepspeed


def add_model_config_args(parser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--model_select', type=str, default='112m')

    return parser


def add_fp16_config_args(parser):
    """Mixed precision arguments."""

    group = parser.add_argument_group('fp16', 'fp16 configurations')


    return parser


def add_training_args(parser):
    """Training arguments."""

    group = parser.add_argument_group('train', 'training configurations')

    group.add_argument('--seed', type=int, default=123,
                       help='random seed')
    group.add_argument('--ckpt_dir', type=str, default='./checkpoints',
                       help='directory for save checkpoint')
    group.add_argument('--workspace', type=str, default='test0',
                       help='workspace directory name')
    group.add_argument('--train_iters', type=int, default=100_000,
                       help='# of iterations for training')
    group.add_argument('--tr_ratio', type=float, default=0.95,
                       help='ratio for training in total dataset')
    group.add_argument('--loss_type', type=str, default='lm_loss',
                       help='cross entropy loss')
    parser.add_argument('--wandb_dir',
                        type=str,
                        default='kg_gpt2_0215',
                        help='for setting wandb')

    # distributed training args
    group.add_argument('--distributed-backend', default='nccl',
                       help='which backend to use for distributed '
                       'training. One of [gloo, nccl]')

    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')

    return parser


def add_evaluation_args(parser):
    """Evaluation arguments."""

    group = parser.add_argument_group('validation', 'validation configurations')

    group.add_argument('--eval_batch_size', type=int, default=128,
                       help='# of batch size for evaluating on each GPU')
    group.add_argument('--load_dir', type=str, default='./checkpoints/test2')
    group.add_argument('--ckpt_id', type=str)

    return parser

def add_text_generate_args(parser):
    """Text generate arguments."""

    group = parser.add_argument_group('Text generation', 'configurations')

    return parser


def add_data_args(parser):
    """Train/valid/test data arguments."""

    group = parser.add_argument_group('data', 'data configurations')
    group.add_argument('--config_train', type=str)

    return parser


def add_preprocessing_args(parser):
    group = parser.add_argument_group('preprocessing', 'data preprocessing')
    group.add_argument('--config_src', type=str)
    group.add_argument('--config_trgt', type=str)
    group.add_argument('--num_process', type=int, default=10)
    group.add_argument('--vocab_train_dir', type=str)
    group.add_argument('--vocab_train_fname', type=str)
    group.add_argument('--vocab_dir', type=str)
    group.add_argument('--nsample', type=int)
    group.add_argument('--vocab_prefix', type=str)

    return parser


def get_ds_args():
    """Parse all the args."""

    parser = argparse.ArgumentParser(description='PyTorch BERT Model')
    parser = add_model_config_args(parser)
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
