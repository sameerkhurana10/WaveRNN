# WaveRNN

Pytorch implementation of Deepmind's WaveRNN model from [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435v1)

This is a fork that implements a standalone version based on the notebooks by fatchord (https://github.com/fatchord/WaveRNN).


<img src="https://raw.githubusercontent.com/fatchord/WaveRNN/master/assets/WaveRNN.png" alt="drawing" width="600px"/>
<img src="https://raw.githubusercontent.com/fatchord/WaveRNN/master/assets/wavernn_alt_model_hrz2.png" alt="drawing" width="600px"/>




# Pretrained models

Trained on LJSpeech:

* Commit: https://github.com/geneing/WaveRNN/commit/fa282a4b7c8c31318b47617688fad3b8ed6856df (https://app.box.com/s/uij3u2x6rxx2m01kbzirr4cto0jclqxe)

After 400 epochs, the quality is quite good.

### Dependencies
* Python 3
* Pytorch v0.5
* Librosa

**Disclaimer** I do not represent or work for Deepmind/Google.
