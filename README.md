# Deep Residual Coupled Prompt Learning for Zero-Shot Sketch-Based Image Retrieval

![Fig.1](DRCPL-Model.png)

## ========= Installation and Requirements =========

- ``` cudatoolkit=11.3.1  ```

- ``` numpy=1.19.3  ```

- ``` python=3.7.16  ```

- ``` pytorch=1.10.0  ```

- ``` torchvision=0.11.0  ```

## Datasets
Please download SBIR datasets from the official websites or Google Drive and `tar -zxvf dataset` to the corresponding directory in `./datasets`. We provide train and test splits for different datasets.

### Sketchy
Please go to the [Sketchy official website](https://sketchy.eye.gatech.edu/), or download the dataset from [Google Drive](https://drive.google.com/file/d/11GAr0jrtowTnR3otyQbNMSLPeHyvecdP/view?usp=sharing).

### TU-Berlin
Please go to the [TU-Berlin official website](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/), or download the dataset from [Google Drive](https://drive.google.com/file/d/12VV40j5Nf4hNBfFy0AhYEtql1OjwXAUC/view?usp=sharing).

### QuickDraw
Please go to the [QuickDraw official website](https://github.com/googlecreativelab/quickdraw-dataset), or download the dataset from [Google Drive](https://drive.google.com/file/d/1EZ8xWRzCi8JcKiFtciD2PwguofC785gK/view?usp=sharing).

## ============== Training ==============

- ``` CUDA_VISIBLE_DEVICES=1 python train.py  ```

## ============== Testing ==============

- ``` CUDA_VISIBLE_DEVICES=1 python test.py  ```
