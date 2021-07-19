# Multi-objective search of robust neural architectures against multiple types of adversarial attacks
## Introduction
Many existing deep learning models are vulnerable to adversarial examples that are imperceptible to humans. To address this issue, various methods have been proposed to design network architectures that are robust to one particular type of adversarial attacks. It is practically impossible, however, to predict beforehand which type of attacks a machine learn model may suffer from. To address this challenge, we propose to search for deep neural architectures that are robust to five types of well-known adversarial attacks using a multi-objective evolutionary algorithm. To reduce the computational cost, a normalized error rate of a randomly chosen attack is calculated as the robustness for each newly generated neural architecture at each generation. All non-dominated network architectures obtained by the proposed method are then fully trained against randomly chosen adversarial attacks and tested on two widely used datasets. Our experimental results demonstrate the superiority of optimized neural architectures found by the proposed approach over state-of-the-art networks that are widely used in the literature in terms of the classification accuracy under different adversarial attacks.
## Install
Using a virtual python environment is encouraged. For example, with Anaconda, you could run `conda create -n moras python==3.7.3 pip` first.
* Supported python versions: 3.7
* Architecture plotting depends on the `graphviz` package, make sure `graphiz` is installed
## Run MORAS
```
  cd CIFAR10\search python rdm_evolution_search_main.py
```
## Citations
If you find the code useful for your research, please consider citing our works.
```
@article{LIU202173,
title = {Multi-objective search of robust neural architectures against multiple types of adversarial attacks},
journal = {Neurocomputing},
volume = {453},
pages = {73-84},
year = {2021},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2021.04.111},
url = {https://www.sciencedirect.com/science/article/pii/S092523122100669X},
author = {Jia Liu and Yaochu Jin}
}
```
## References
* **NSGANET** Zhichao Lu, Ian Whalen, Yashesh Dhebar, Kalyanmoy Deb, Erik Goodman, Wolfgang Banzhaf, Vishnu Naresh Boddeti. "NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm". In Proceedings of the Genetic and Evolutionary Computation Conference, 2019
