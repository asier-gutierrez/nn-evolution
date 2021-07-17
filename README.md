# Neural Network Learning Monitoring Without A Validation Set
This repository contains the code for our paper "Persistent Homology Captures the Generalization of Neural Networks Without A Validation Set".

## Computational Requirements
This code requires large Random Access Memory (RAM). We used a machine of 1.5TB of RAM and 128 physical processor cores.

We suggest using either a Cloud Computing Machine (Amazon Web Services, Microsoft Azure, Google Cloud, IBM Cloud or similar) or reducing the network capacity.

##  Execution
Install the requirements. We suggest using conda.

```
conda create --name <env> --file requirements.txt
```

To execute the experiments by batch:
```
python learning/batch/(cifar100_cnn.py|cifar10_cnn.py|cifar100_mlp.py|cifar10_mlp.py|mnist_mlp.py|reuters_mlp.py)
```
_You might need to download some of the Keras datasets used_

Configuration of experiments can be changed in the `learning/conf` folder and analysis/plotting code can be found in the `learning/analysis` folder.

Executions by epoch can be performed and are faster as they do not need to compute NN graph and Persistent Homology for each batch. 
However, it was discontinued: as a suggestion you should disable persisting the network graphs in memory (it takes 800GB aproximately).
Note that the visualizations for epoch executions were discontinued too, they can be performed but they might look a little bit ugly (not the results but the matplotlib visualizations).

## Citing
See our pre-print on arXiv: https://arxiv.org/abs/2106.00012

Cite our paper:
```
@misc{gutierrez-fandino2021nnph,
      title={Persistent Homology Captures the Generalization of Neural Networks Without A Validation Set}, 
      author={Asier Gutiérrez-Fandiño and David Pérez-Fernández and Jordi Armengol-Estapé and Marta Villegas},
      year={2021},
      eprint={2106.00012},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
