# Kuzushiji-49 Character Recognition

## Summary
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Training](#training)
- [Evaluation](#evaluation)
  - [Accuracy](#accuracy)
  - [ROC Curve and AUC](#roc-curve-and-auc)
  - [Multiclass ROC/AUC Analysis](#multiclass-roc-auc-analysis)
- [Discussion](#discussion)


## Introduction
This project consists in training 12 [Neural Networks and Convolutional Neural Networks](https://youtube.com/playlist?list=PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU), in order to compare their performance and gain insight about how certain hyperparameters - such as the number of layers, filters, optimizers, loss functions - impact in the performance, by using metrics such as the overall accuracy and, mainly the ROC/AUC analysis.

## Dataset
The dataset used is the Kuzushiji-49 dataset, that comes as one of the datasets within the Kuzushiji-MNIST dataset. It consists of 232365 images for training and 38547 for evaluation, with 49 classes (for 49 [Hentaigana](https://en.wikipedia.org/wiki/Hentaigana) characters) as the name of the dataset suggests.

Every sample is a 28x28 gray-scale image, saved in numpy compressed files (npz format). The figure below shows 75 random samples from the dataset.

<p align="center">
  <img align="center" alt="75 Random Samples" src="/examples/dark/75_random_samples.png#gh-light-mode-only">
  <img align="center" alt="75 Random Samples" src="/examples/light/75_random_samples.png#gh-dark-mode-only">
</p>

As explained in the Kaggle page for the dataset, the Kuzushiji characters from the dataset were extracted from old japanese media, and as such, they can show different variations for the same character. That means that many samples from the same class can look very different, making it hard even for experts to distinguish them, and also for training models to learn this dataset well. The figure below ilustrates this situation.

<p align="center">
  <img align="center" alt="5 Classes Train Samples" src="/examples/dark/5_classes_train_samples.png#gh-light-mode-only">
  <img align="center" alt="5 Classes Train Samples" src="/examples/light/5_classes_train_samples.png#gh-dark-mode-only">
</p>

And besides having samples with variations for the same class, there are also less common characters in the japanese literature, than others. To translate this reality to the dataset, the unbalance between the amount of samples was kept, as shown in the figure below.

<p align="center">
  <img align="center" alt="Classes Unbalance" src="/examples/dark/classes_inbalance_train_samples.png#gh-light-mode-only">
  <img align="center" alt="Classes Unbalance" src="/examples/light/classes_inbalance_train_samples.png#gh-dark-mode-only">
</p>

For more details, check the [Kaggle page](https://www.kaggle.com/anokas/kuzushiji) for the dataset.

## Models
In the [original article](https://arxiv.org/pdf/1812.01718.pdf) the authors showed the classification results for different models (including ResNet models) on the MNIST and the Kuzushiji-49 datasets, aiming to demonstrate how much more difficult the Kusuzhiji-49 dataset can be. For this personal project of mine, I decided to try different architectures for neural networks (NNs) and convolutional neural networks (CNNs), since this is my very first Keras project.

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

Both NNs and CNNs models share the same idea. The first 4 models aims to compare how the number of layers/filters and different optimizers, SGD (Stochastic Gradient Descent) and Adam, impact on the models performance. And the last 2 models, are more dense networks, whose goal is to see how much better a model gets, if at all, with many more layers/filters, and also if [Batch Normalization](https://www.youtube.com/watch?v=dXB-KQYkzNU&t=7s) plays an important role in the learning process.

## Training
Every model was trained for 10 epochs, using batches with 128 samples, with each epoch having the same 30% of the dataset separated for validation (since shuffling the validation set in between the epochs didn't seem to make any difference).

When the training of a model is finished, the trained weights are saved as .h5 files, for simple loading afterwards, avoiding having to train the models all over again (which takes a long time, specially for the CNNs).

## Evaluation
As explained in the section [Dataset](#dataset), the dataset is a bit unbalanced (with some classes only having ~400 samples, way less than 6000 samples for other classes). Because of this, it's not fair to evaluate these models based purely on accuracy. It is necessary to use a multiclass ROC/AUC analysis to fully evaluate the learning performance of each model, which will be done later on, after the accuracy analysis.

### Accuracy
The overall accuracy obtained by each model is shown below.

<p align="center">
  <img align="center" alt="Accuracies" src="/examples/dark/results/accuracies.svg#gh-light-mode-only">
  <img align="center" alt="Accuracies" src="/examples/light/results/accuracies.svg#gh-dark-mode-only">
</p>

Based in it, we can draw some conclusions about the models:
- Every CNN achieved superior accuracy than any NN;
- The models that used the SGD algorithm as the optimizer, both CNN and NN, achieved lower accuracy than it's counterparts that used Adam instead;
- The CNN_dense_batch_norm model obtained very similar accuracy to the other simpler CNN models, even though it is much more expensive to train;
- However the NN_dense_batch_norm managed to get better accuracy when compared to every other NN model;

### ROC Curve and AUC
As mentioned previously, the dataset is (purposely) unbalanced, with some classes having way less samples than the others. Therefore evaluating the performance by only measuring the accuracy is not enough to conclude which is the best model.<br>
The more commonly used methodoly for multiclass classification problems is to, from the confusion matrix, extract metrics such as true positive rate (sensibility) and false positive rate (1-specificity) to plot the ROC Curve for every model, and from it, calculate the Area Under the Curve (AUC), where the model with highest AUC value wins.<br>
However, the way it was just described, this methodology only applies to binary classification problems, where **every model has only one ROC Curve** (and consequently one AUC value associated with it), but the Kuzushiji-49 dataset presents a 49-th classification problem, where **every model has 49 ROC Curves**, having 49 AUC values associated with every model. To solve this, the ROC/AUC analysis needs to be expanded.

### Multiclass ROC/AUC Analysis
The idea here is not very different than what was first introduced. The only difference is that in order to calculate the true and false positive rate values, needed to plot the ROC Curve, the "one-vs-all" strategy is required (there are others, e.g. the "one-vs-one"), where one class is considered as the positive class and all the rest as the negative class. This way, it's possible to have a ROC Curve associated with every class for every model. But that's very hard to use to get any insight, just take a look at the figure below.

<p align="center">
  <img align="center" alt="ROC Nightmare" src="/examples/dark/results/ROC_nightmare.svg#gh-light-mode-only">
  <img align="center" alt="ROC Nightmare" src="/examples/light/results/ROC_nightmare.svg#gh-dark-mode-only">
</p>

It's possible to see that some curves are better than the others (the highest and most to the left curves - nearer to the "ROC heaven") however it is very hard to draw any assumptions or conclusions from it. The next option would be to calculate the AUC values for every ROC Curve associated with every model.<br>
This analysis can be interesting to see how well the models learned every class, leading to some insights about how hard a given class can be to properly learn. The comparisions that are gonna be made consists in:
- Highest accuracy NN with Adam optimizer vs Highest accuracy NN with SGD optimizer;
- Highest accuracy NN with Adam optimizer vs Highest accuracy NN with SGD optimizer;
- NN_dense vs NN_dense_batch_norm;
- CNN_dense vs CNN_dense_batch_norm;
- Highest accuracy NN vs Highest accuracy CNN;

#### NN_2_adam vs NN_2_sgd
<p align="center">
  <img align="center" alt="NN_2_adam_auc_x_NN_2_sgd_auc" src="/examples/dark/results/NN_2_adam_auc_x_NN_2_sgd_auc.svg#gh-light-mode-only">
  <img align="center" alt="NN_2_adam_auc_x_NN_2_sgd_auc" src="/examples/light/results/NN_2_adam_auc_x_NN_2_sgd_auc.svg#gh-dark-mode-only">
</p>

Bla bla bla

#### CNN_1_adam vs CNN_2_sgd
<p align="center">
  <img align="center" alt="CNN_1_adam_auc_x_CNN_2_sgd_auc" src="/examples/dark/results/CNN_1_adam_auc_x_CNN_2_sgd_auc.svg#gh-light-mode-only">
  <img align="center" alt="CNN_1_adam_auc_x_CNN_2_sgd_auc" src="/examples/light/results/CNN_1_adam_auc_x_CNN_2_sgd_auc.svg#gh-dark-mode-only">
</p>

Bla bla bla

#### NN_dense vs NN_dense_batch_norm
<p align="center">
  <img align="center" alt="NN_dense_auc_x_NN_dense_batch_norm_auc" src="/examples/dark/results/NN_dense_auc_x_NN_dense_batch_norm_auc.svg#gh-light-mode-only">
  <img align="center" alt="NN_dense_auc_x_NN_dense_batch_norm_auc" src="/examples/light/results/NN_dense_auc_x_NN_dense_batch_norm_auc.svg#gh-dark-mode-only">
</p>

Bla bla bla

#### CNN_dense vs CNN_dense_batch_norm
<p align="center">
  <img align="center" alt="CNN_dense_auc_x_CNN_dense_batch_norm_auc" src="/examples/dark/results/CNN_dense_auc_x_CNN_dense_batch_norm_auc.svg#gh-light-mode-only">
  <img align="center" alt="CNN_dense_auc_x_CNN_dense_batch_norm_auc" src="/examples/light/results/CNN_dense_auc_x_CNN_dense_batch_norm_auc.svg#gh-dark-mode-only">
</p>

Bla bla bla

#### NN_dense_batch_norm vs CNN_dense_batch_norm
<p align="center">
  <img align="center" alt="NN_dense_batch_norm_auc_x_CNN_dense_batch_norm_auc" src="/examples/dark/results/NN_dense_batch_norm_auc_x_CNN_dense_batch_norm_auc.svg#gh-light-mode-only">
  <img align="center" alt="NN_dense_batch_norm_auc_x_CNN_dense_batch_norm_auc" src="/examples/light/results/NN_dense_batch_norm_auc_x_CNN_dense_batch_norm_auc.svg#gh-dark-mode-only">
</p>


---------------------------------------------------------------------
That's why it's better to use the Weighted Average AUC.

### Weighted Average AUC
Finally, the 
----------------------------------------------------------------------



<p align="center">
  <img align="center" alt="Classes Inbalance" src="/examples/dark/results/auc_weighted_avg.svg#gh-light-mode-only">
  <img align="center" alt="Classes Inbalance" src="/examples/light/results/auc_weighted_avg.svg#gh-dark-mode-only">
</p>

<p align="center">
  <img align="center" alt="Classes Inbalance" src="/examples/dark/results/scatter_accuracy_auc.svg#gh-light-mode-only">
  <img align="center" alt="Classes Inbalance" src="/examples/light/results/scatter_accuracy_auc.svg#gh-dark-mode-only">
</p>

## Discussion

TODO - Talk about using ResNets, for reduced computational cost -> less time to train
