# MERTools

Correspondence to: 

  - Zheng Lian: lianzheng2016@ia.ac.cn
  - Licai Sun: sunlicai2019@ia.ac.cn

## Environment
```shell
conda env create -f environment.yml
```
- If raise errors about "OSError: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory", please run "pip install -U torch torchaudio --no-cache-dir"
- If your Cuda version is low (such as 10.2), please check the install instructions for pytorch-relate packages in "https://pytorch.org/get-started/previous-versions"


## MER2023 Dataset

To download the dataset, please fill out an [EULA](https://drive.google.com/file/d/1LOW2e6ZuyUjurVF0SNPisqSh4VzEl5lN) and send it to our official email address merchallenge.contact@gmail.com or lianzheng2016@ia.ac.cn. It requires participants to use this dataset only for academic research and not to edit or upload samples to the Internet.


## MER2023 Baseline

[**MER 2023: Multi-label Learning, Modality Robustness, and Semi-Supervised Learning**](https://dl.acm.org/doi/pdf/10.1145/3581783.3612836)<br>
Zheng Lian, Haiyang Sun, Licai Sun, Jinming Zhao, Ye Liu, Bin Liu, Jiangyan Yi, Meng Wang, Erik Cambria, Guoying Zhao, Björn W. Schuller, Jianhua Tao<br>

Please cite our paper if you find our work useful for your research:

```tex
@inproceedings{lian2023mer,
  title={Mer 2023: Multi-label learning, modality robustness, and semi-supervised learning},
  author={Lian, Zheng and Sun, Haiyang and Sun, Licai and Chen, Kang and Xu, Mngyu and Wang, Kexin and Xu, Ke and He, Yu and Li, Ying and Zhao, Jinming and others},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={9610--9614},
  year={2023}
}
```
paper: https://dl.acm.org/doi/pdf/10.1145/3581783.3612836

code: see the **./MER2023** folder



## MERBench

[**MERBench: A Unified Evaluation Benchmark for Multimodal Emotion Recognition**](https://arxiv.org/pdf/2401.03429)<br>
Zheng Lian, Licai Sun, Yong Ren, Hao Gu, Haiyang Sun, Lan Chen, Bin Liu, Jianhua Tao<br>

Please cite our paper if you find our work useful for your research:

```tex
@article{lian2023mer,
  title={MERBench: A Unified Evaluation Benchmark for Multimodal Emotion Recognition},
  author={Lian, Zheng and Sun, Licai and Ren, Yong and Gu, Hao and Sun, Haiyang and Chen, Lan and Liu, Bin and Tao, Jianhua},
  journal={arXiv:2401.03429},
  year={2024}
}
```

paper: https://arxiv.org/pdf/2401.03429

code: see the **./MERBench** folder



