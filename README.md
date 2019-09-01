## Description
I am researching end-to-end ASR, such as CTC, Transducer and so on. There is a lot of variants of LSTM proposed for ASR task. In the [paper](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43905.pdf), the lstm with projection layer gets better performance. But the lstmp isn't supported by Pytorch, so I implement this custom LSTM according to [this tutorial](https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/). I hope it can helps other researchers. 

## References

[Long Short-Term Memory Recurrent Neural Network Architectures
for Large Scale Acoustic Modeling](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43905.pdf)

[Optimizing CUDA Recurrent Neural Networks with TorchScript](https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/))
