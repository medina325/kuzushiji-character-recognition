# Kuzushiji-49 Character Recognition

## Summary
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Training](#training)
- [Evaluation](#evaluation)
- [Discussion](#discussion)


## Introduction
This project consists in training several [Neural Networks and Convolutional Neural Networks](https://youtube.com/playlist?list=PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU), in order to compare their performance and gain insight about how certain hyperparameters - such as the number of layers, filters, optimizers, loss functions - impact in the performance, by using metrics such as the overall accuracy and, mainly the ROC analysis.

## Dataset
The dataset used is the Kuzushiji-49 dataset, that comes as one of the datasets within the Kuzushiji-MNIST dataset. It consists in 232365 images for training and 38547 for evaluation, with 49 classes (for 49 [Hentaigana](https://en.wikipedia.org/wiki/Hentaigana) characters) as the name of the dataset suggests.

Every sample is a 28x28 gray-scale image, saved in numpy compressed files (npz format). The figure below shows 75 random samples from the dataset.

<p align="center">
  <img align="center" alt="75 Random Samples" src="/plots/dark/75_random_samples.png#gh-light-mode-only">
  <img align="center" alt="75 Random Samples" src="/plots/light/75_random_samples.png#gh-dark-mode-only">
</p>

As explained in the Kaggle page for the dataset, Kuzushiji characters have different variations for the same character. That means that many samples from the same class can look very different, making it hard even for experts to distinguish them, and also for training models to learn this dataset well. The figure below ilustrates this situation.

<p align="center">
  <img align="center" alt="5 Classes Train Samples" src="/plots/dark/5_classes_train_samples.png#gh-light-mode-only">
  <img align="center" alt="5 Classes Train Samples" src="/plots/light/5_classes_train_samples.png#gh-dark-mode-only">
</p>

And besides having samples with variations for the same class, there are also less common characters in the japanese literature, than others. To translate this reality to the dataset, the inbalance between the amount of samples was kept, as shown in the figure below.

<p align="center">
  <img align="center" alt="Classes Inbalance" src="/plots/dark/classes_inbalance_train_samples.png#gh-light-mode-only">
  <img align="center" alt="Classes Inbalance" src="/plots/light/classes_inbalance_train_samples.png#gh-dark-mode-only">
</p>

For more details, check the [Kaggle page](https://www.kaggle.com/anokas/kuzushiji) for the dataset.

## Models
In the [original article](https://arxiv.org/pdf/1812.01718.pdf) the authors showed the classification results for different models (including ResNet models) on the MNIST and the Kuzushiji-49 datasets, aiming to demonstrate how much more difficult the Kusuzhiji-49 dataset can be. For this personal project of mine, I decided to try different architectures for neural networks (NNs) and convolutional neural networks (CNNs), since this is the first Keras project I am developing.

The models consists in 6 NNs and 6 CNNs:

- NN_1_sgd
- NN_2_sgd
- NN_1_adam
- NN_2_adam
- NN_dense
- NN_dense_batch_form
---
- CNN_1_sgd
- CNN_2_sgd
- CNN_1_adam
- CNN_2_adam
- CNN_dense
- CNN_dense_batch_form

Both NNs and CNNs models share the same idea. The first 4 models aims to compare how the number of layers/filters and different optimizers, SGD (Stochastic Gradient Descent) and Adam, impact on the models performance. And the last 2 models, are more dense networks, whose goal is to see how much better a model gets, if at all, with many more layers/filters, and also if [Batch Normalization](https://www.youtube.com/watch?v=dXB-KQYkzNU&t=7s) plays an important part in the learning process.

## Training
Every model was trained for 10 epochs, using batches with 128 samples, with each epoch having the same 30% of the dataset separated for validation (since shuffling the validation set in between the epochs didn't seem to make any difference).

When the training of a model is finished, the trained weights are saved as .h5 files, for simple loading afterwards, avoiding having to train the models all over again (which takes a long time, specially for the CNNs).

## Evaluation
As explained in the section [Dataset](#dataset), the dataset is a bit unbalanced (with some classes only having ~400 samples, way less than 6000 samples for other classes). Because of this, it's not fair to evaluate these models based purely on accuracy. It is necessary to use a multiclass ROC analysis .............

TODO - show / analyse results

## Discussion

TODO - Talk about using ResNets, for reduced computational cost -> less time to train
