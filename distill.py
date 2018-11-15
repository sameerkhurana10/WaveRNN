"""Trainining script for reduced WaveRNN vocoder

usage: distill.py [options]

options:
    --data-root=<dir>            Directory contains preprocessed features.
    --checkpoint-dir=<dir>       Directory where to save model checkpoints [default: checkpoints].
    --output-dir=<dir>           Output Directory [default: model_outputs]
    --hparams=<params>           Hyper parameters [default: ].
    --preset=<json>              Path of preset parameters (json).
    --checkpoint=<path>          Restore model from checkpoint path if given.
    --restore-parts=<path>       Restore part of the model.
    --log-event-path=<name>      Log event path.
    --reset-optimizer            Reset optimizer.
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
from models import *

class AudioDataset(Dataset):
    def __init__(self, ids, path):
        self.path = path
        self.metadata = ids

    def __getitem__(self, index):
        file = self.metadata[index]
        m = np.load(f'{self.path}/mel/{file}.npy')
        x = np.load(f'{self.path}/quant/{file}.npy')
        return m, x

    def __len__(self):
        return len(self.metadata)


def collate(batch):
    pad = 2
    mel_win = seq_len // dsp.hop_length + 2 * pad
    max_offsets = [x[0].shape[-1] - (mel_win + 2 * pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + pad) * dsp.hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] \
            for i, x in enumerate(batch)]

    coarse = [x[1][sig_offsets[i]:sig_offsets[i] + seq_len + 1] \
              for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    coarse = np.stack(coarse).astype(np.int64)

    mels = torch.FloatTensor(mels)
    coarse = torch.LongTensor(coarse)

    x_input = 2 * coarse[:, :seq_len].float() / (2**hparams.bits - 1.) - 1.
    y_coarse = coarse[:, 1:]

    return x_input, mels, y_coarse


def train(teacher, student, optimiser, epochs, step, lr=1e-4):

    for p in optimiser.param_groups:
        p['lr'] = lr
    criterion = nn.L1Loss().to(device)

    trn_loader = DataLoader(dataset, collate_fn=collate, batch_size=hparams.batch_size,
                            num_workers=hparams.num_workers, shuffle=True, pin_memory=(not no_cuda))
    iters = len(trn_loader)

    for e in range(epochs):

        running_loss = 0.
        val_loss = 0.
        start = time.time()

        #with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
        #with torch.autograd.detect_anomaly():
        #with torch.autograd.profiler.emit_nvtx(enabled=False):
        for i, (x, m, y) in enumerate(trn_loader) :

            x, m, y = x.to(device), m.to(device), y.to(device)
            if no_cuda:
                y_teacher_hat = teacher(x, m)
            else:
                y_teacher_hat = torch.nn.parallel.data_parallel(teacher, (x, m))
            y_teacher_hat = y_teacher_hat.transpose(1, 2).unsqueeze(-1)

            for kiter in range(5):
                optimiser.zero_grad()
                if no_cuda:
                    y_student_hat = student(x, m)
                else:
                    y_student_hat = torch.nn.parallel.data_parallel(student, (x, m))

                y_student_hat = y_student_hat.transpose(1, 2).unsqueeze(-1)

                loss = 1000.*criterion(y_teacher_hat, y_student_hat)

                loss.backward()
                optimiser.step()
                running_loss += loss.item()

                avg_loss = running_loss / (i + 1)

                step += 1
                speed = step / (time.time() - start)

                k = step // 1000
                print('Epoch: %i/%i -- Batch: %i/%i -- Loss: %.3f -- Speed: %.2f steps/sec -- Step: %ik '%
                        (e + 1, epochs, i + 1, iters, avg_loss, speed, k))

        #print(prof.table(sort_by='cuda_time'))
        #prof.export_chrome_trace(f'{output_path}/chrome_trace')

        torch.save(student.state_dict(), STUDENT_MODEL_PATH)
        np.save(checkpoint_step_path, step)
        if e % 20 == 0:
            generate(student, e, data_root, output_path, test_ids)
        print(' <saved>')


def generate(model, epoch, data_root, output_path, test_ids, samples=3):
    test_mels = [np.load(f'{data_root}/mel/{dataset_id}.npy') for dataset_id in test_ids[:samples]]
    ground_truth = [np.load(f'{data_root}/quant/{dataset_id}.npy') for dataset_id in test_ids[:samples]]
    os.makedirs(f'{output_path}/{epoch}', exist_ok=True)

    for i, (gt, mel) in enumerate(zip(ground_truth, test_mels)):
        print('\nGenerating: %i/%i' % (i+1, samples))
        gt = 2 * gt.astype(np.float32) / (2**hparams.bits - 1.) - 1.
        dsp.save_wav(gt, f'{output_path}/{epoch}/student_target_{i}.wav')
        output = model.generate(mel)
        dsp.save_wav(output, f'{output_path}/{epoch}/student_generated_{i}.wav')


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    output_path = args["--output-dir"]
    checkpoint_path = args["--checkpoint"]
    checkpoint_restore_parts = args["--restore-parts"]
    preset = args["--preset"]

    data_root = args["--data-root"]
    if data_root is None:
        data_root = os.join(os.dirname(__file__), "data", "ljspeech")

    log_event_path = args["--log-event-path"]
    reset_optimizer = args["--reset-optimizer"]
    no_cuda = args["--no-cuda"]

    device = torch.device("cpu" if no_cuda else "cuda")

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "WaveRNN"
    #print(hparams_debug_string())
    hparams.distill=True

    dsp = DSP(hparams)
    hparams.hop_length=dsp.hop_length
    seq_len = dsp.hop_length * 5
    step = 0

    os.makedirs(f'{checkpoint_dir}/', exist_ok=True)

    MODEL_PATH = f'{checkpoint_dir}/model.pyt'
    STUDENT_MODEL_PATH = f'{checkpoint_dir}/student_model.pyt'
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

    teacher = Model(device, rnn_dims=hparams.rnn_dims, fc_dims=hparams.fc_dims, bits=hparams.bits, pad=hparams.pad,
                  upsample_factors=hparams.upsample_factors, feat_dims=hparams.feat_dims,
                  compute_dims=hparams.compute_dims, res_out_dims=hparams.res_out_dims, res_blocks=hparams.res_blocks).to(device)

    assert(os.path.exists(MODEL_PATH))
    print("\t\t Loading saved state")
    teacher.load_state_dict(torch.load(MODEL_PATH))
    for param in teacher.parameters():
        param.requires_grad = False

    student = Model(device, rnn_dims=hparams.student_rnn_dims, fc_dims=hparams.student_fc_dims, bits=hparams.bits,
                    pad=hparams.pad, upsample_factors=hparams.student_upsample_factors,
                    feat_dims=hparams.student_feat_dims, compute_dims=hparams.student_compute_dims,
                    res_out_dims=hparams.student_res_out_dims, res_blocks=hparams.student_res_blocks).to(device)
    #student.load_state_dict(torch.load(STUDENT_MODEL_PATH))

    optimiser = optim.Adam(student.parameters())
    train(teacher, student, optimiser, epochs=hparams.epochs, step=step, lr=hparams.lr)

    generate(student, step, data_root, output_path, test_ids)

