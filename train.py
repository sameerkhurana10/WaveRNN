"""Trainining script for WaveRNN vocoder

usage: train.py [options]

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


class ResBlock(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(nn.Module):
    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims):
        super().__init__()
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=5, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks) :
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers: x = f(x)
        x = self.conv_out(x)
        return x


class Stretch2d(nn.Module):
    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)


class UpsampleNetwork(nn.Module):
    def __init__(self, feat_dims, upsample_scales, compute_dims,
                 res_blocks, res_out_dims, pad):
        super().__init__()
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales :
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1. / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers:
            m = f(m)
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)


class Model(nn.Module):
    def __init__(self, rnn_dims, fc_dims, bits, pad, upsample_factors,
                 feat_dims, compute_dims, res_out_dims, res_blocks):
        super().__init__()
        self.n_classes = 2**bits
        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 4
        self.upsample = UpsampleNetwork(feat_dims, upsample_factors, compute_dims,
                                        res_blocks, res_out_dims, pad)
        self.I = nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims)
        self.rnn1 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn2 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)
        self.fc1 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, self.n_classes)
        display.num_params(self)

    def forward(self, x, mels):
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()

        bsize = x.size(0)
        h1 = torch.zeros(1, bsize, self.rnn_dims).to(device)
        h2 = torch.zeros(1, bsize, self.rnn_dims).to(device)
        mels, aux = self.upsample(mels)

        aux_idx = [self.aux_dims * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]

        x = torch.cat([x.unsqueeze(-1), mels, a1], dim=2)
        x = self.I(x)
        res = x
        x, _ = self.rnn1(x, h1)

        x = x + res
        res = x
        x = torch.cat([x, a2], dim=2)
        x, _ = self.rnn2(x, h2)

        x = x + res
        x = torch.cat([x, a3], dim=2)
        x = F.relu(self.fc1(x))

        x = torch.cat([x, a4], dim=2)
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=-1)

    def preview_upsampling(self, mels) :
        mels, aux = self.upsample(mels)
        return mels, aux

    @staticmethod
    def _round_up(num, divisor):
        return num + divisor - (num%divisor)

    def _batch_mels(self, mel):
        n_hops = mel.shape[1]
        n_pad = self._round_up(n_hops, hparams.batch_size_gen) - n_hops
        mel_padded = np.pad(mel, [(0, 0), (0, n_pad)], 'constant', constant_values=0)
        mel_batched = mel_padded.reshape([mel.shape[0], hparams.batch_size_gen, -1]).swapaxes(0, 1)
        pad_length = n_pad*dsp.hop_length      # ToDo: remove dependency on dsp
        return mel_batched, pad_length

    @staticmethod
    def _unbatch_sound(x, pad_length):
        y = x.flatten()[:-pad_length]
        return y

    def generate(self, mel, save_path):
        self.eval()
        mels, pad_length = self._batch_mels(mel)

        bsize = mels.shape[0]
        output = []
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        with torch.no_grad():
            start = time.time()
            x = torch.zeros(bsize, 1).to(device)
            h1 = torch.zeros(bsize, self.rnn_dims).to(device)
            h2 = torch.zeros(bsize, self.rnn_dims).to(device)

            mels = torch.FloatTensor(mels).to(device)
            mels, aux = self.upsample(mels)

            aux_idx = [self.aux_dims * i for i in range(5)]
            a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
            a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
            a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
            a4 = aux[:, :, aux_idx[3]:aux_idx[4]]

            seq_len = mels.size(1)

            for i in range(seq_len):

                m_t = mels[:, i, :]
                a1_t = a1[:, i, :]
                a2_t = a2[:, i, :]
                a3_t = a3[:, i, :]
                a4_t = a4[:, i, :]

                x = torch.cat([x, m_t, a1_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)

                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)

                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.fc1(x))

                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                posterior = F.softmax(x, dim=1)
                distrib = torch.distributions.Categorical(posterior)
                sample = 2 * distrib.sample().float() / (self.n_classes - 1.) - 1.
                output.append(sample)
                x = sample.unsqueeze(-1).to(device)
                if i % 100 == 0 :
                    speed = int((i + 1) / (time.time() - start))
                    print('%i/%i -- Speed: %i samples/sec'%(i + 1, seq_len, speed))
        output = torch.stack(output).cpu().numpy()
        output = self._unbatch_sound(output, pad_length)
        if save_path:
            dsp.save_wav(output, save_path)
        self.train()
        return output

    def get_gru_cell(self, gru) :
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell


def train(model, optimiser, epochs, batch_size, classes, seq_len, step, lr=1e-4):

    for p in optimiser.param_groups:
        p['lr'] = lr
    criterion = nn.NLLLoss().to(device)

    for e in range(epochs):
        trn_loader = DataLoader(dataset, collate_fn=collate, batch_size=hparams.batch_size,
                                num_workers=hparams.num_workers, shuffle=True, pin_memory=(not no_cuda))

        running_loss = 0.
        val_loss = 0.
        start = time.time()
        running_loss = 0.

        iters = len(trn_loader)

        with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
        #with torch.autograd.profiler.emit_nvtx(enabled=False):
            for i, (x, m, y) in enumerate(trn_loader) :

                x, m, y = x.to(device), m.to(device), y.to(device)

                if no_cuda:
                    y_hat = model(x, m)
                else:
                    y_hat = torch.nn.parallel.data_parallel(model, (x, m))

                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
                y = y.unsqueeze(-1)
                loss = criterion(y_hat, y)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                running_loss += loss.item()

                speed = (i + 1) / (time.time() - start)
                avg_loss = running_loss / (i + 1)

                step += 1

                k = step // 1000
                print('Epoch: %i/%i -- Batch: %i/%i -- Loss: %.3f -- Speed: %.2f steps/sec -- Step: %ik '%
                        (e + 1, epochs, i + 1, iters, avg_loss, speed, k))

                break   #TODO: remove
        #print(prof.table(sort_by='cuda_time'))
        #prof.export_chrome_trace(f'{output_path}/chrome_trace')

        torch.save(model.state_dict(), MODEL_PATH)
        np.save(checkpoint_step_path, step)
        if e % 10 == 0:
            generate(e, data_root, output_path, test_ids)
        print(' <saved>')


def generate(epoch, data_root, output_path, test_ids, samples=3):
    test_mels = [np.load(f'{data_root}/mel/{dataset_id}.npy') for dataset_id in test_ids[:samples]]
    ground_truth = [np.load(f'{data_root}/quant/{dataset_id}.npy') for dataset_id in test_ids[:samples]]
    os.makedirs(f'{output_path}/{epoch}', exist_ok=True)

    for i, (gt, mel) in enumerate(zip(ground_truth, test_mels)):
        print('\nGenerating: %i/%i' % (i+1, samples))
        gt = 2 * gt.astype(np.float32) / (2**hparams.bits - 1.) - 1.
        dsp.save_wav(gt, f'{output_path}/{epoch}/target_{i}.wav')
        model.generate(mel, f'{output_path}/{epoch}/generated_{i}.wav')


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

