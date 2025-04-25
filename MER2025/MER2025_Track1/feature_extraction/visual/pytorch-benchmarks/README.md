### pytorch-benchmark

Some scripts for validating models on common benchmarks. Assumes at least Python3 and PyTorch 4.0.


### Supported datasets:

* **ImageNet** (this is essentially just a cut-down version of the [official example](https://github.com/pytorch/examples/tree/master/imagenet))
* **Fer2013** - A dataset of greyscale faces labelled with emotions.



### References

**ImageNet**: [paper](https://arxiv.org/abs/1409.0575)

```
@article{ILSVRC15,
Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
Title = {{ImageNet Large Scale Visual Recognition Challenge}},
Year = {2015},
journal   = {International Journal of Computer Vision (IJCV)},
doi = {10.1007/s11263-015-0816-y},
volume={115},
number={3},
pages={211-252}
}
```

**FER2013**: [paper](https://arxiv.org/abs/1307.0414)

```
@inproceedings{goodfellow2013challenges,
  title={Challenges in representation learning: A report on three machine learning contests},
  author={Goodfellow, Ian J and Erhan, Dumitru and Carrier, Pierre Luc and Courville, Aaron and Mirza, Mehdi and Hamner, Ben and Cukierski, Will and Tang, Yichuan and Thaler, David and Lee, Dong-Hyun and others},
  booktitle={International Conference on Neural Information Processing},
  pages={117--124},
  year={2013},
  organization={Springer}
}
```

