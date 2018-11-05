"""Synthesis script for WaveRNN vocoder

usage: synthesize.py [options] <mel_input.npy>

options:
    --checkpoint-dir=<dir>       Directory where model checkpoint is saved [default: checkpoints].
    --output-dir=<dir>           Output Directory [default: model_outputs]
    --hparams=<params>           Hyper parameters [default: ].
    --preset=<json>              Path of preset parameters (json).
    --checkpoint=<path>          Restore model from checkpoint path if given.
    --speaker-id=<N>             Use specific speaker of data in case for multi-speaker datasets.
    --no-cuda                    Don't run on GPU
    -h, --help                   Show this help message and exit
"""
import os
import pickle
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from docopt import docopt
from torch import optim
from torch.utils.data import Dataset, DataLoader

import hparams
from hparams import hparams
import utils.display as display
from utils.dsp import DSP




if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    output_path = args["--output-dir"]
    checkpoint_path = args["--checkpoint"]
    preset = args["--preset"]
    no_cuda = args["--no-cuda"]

    device = torch.device("cpu" if no_cuda else "cuda")

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "WaveRNN"

    file_name = args['<mel_input.npy>']

    dsp = DSP(hparams)
    seq_len = dsp.hop_length * 5
    step = 0

    os.makedirs(f'{checkpoint_dir}/', exist_ok=True)

    MODEL_PATH = f'{checkpoint_dir}/model.pyt'
    #data_root = f'data/'
    checkpoint_step_path = f'{checkpoint_dir}/model_step.npy'
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(data_root, 'dataset_ids.pkl'), 'rb') as f:
        dataset_ids = pickle.load(f)
    test_ids = dataset_ids[-50:]
    dataset_ids = dataset_ids[:-50]

    dataset = AudioDataset(dataset_ids, data_root)
    data_loader = DataLoader(dataset, collate_fn=collate, batch_size=hparams.batch_size,
                             num_workers=hparams.num_workers, shuffle=True)

    model = Model(rnn_dims=hparams.rnn_dims, fc_dims=hparams.fc_dims, bits=hparams.bits, pad=hparams.pad,
                  upsample_factors=hparams.upsample_factors, feat_dims=hparams.feat_dims,
                  compute_dims=hparams.compute_dims, res_out_dims=hparams.res_out_dims, res_blocks=hparams.res_blocks).to(device)

    if not os.path.exists(MODEL_PATH):
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        print("\t\t Loading saved state")
        model.load_state_dict(torch.load(MODEL_PATH))

    optimiser = optim.Adam(model.parameters())
    train(model, optimiser, epochs=hparams.epochs, batch_size=hparams.batch_size, classes=2**hparams.bits,
          seq_len=seq_len, step=step, lr=hparams.lr)

    generate(step, data_root, output_path, test_ids)

