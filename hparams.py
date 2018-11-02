
import tensorflow as tf

# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    name="WaveRNN",
    batch_size=32,
    num_workers=8,
    rnn_dims=(64, 32, 16),
    fc_dims=(128, 64, 32),
    pad=2,
    upsample_factors=(5, 5, 11),
    feat_dims=80,
    compute_dims=64,
    res_out_dims=64,
    res_blocks=10,
    epochs=1000,
    lr=1.e-4,
    bits=10
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)