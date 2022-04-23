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
  <img align="center" alt="Classes Unbalance" src="/examples/dark/classes_unbalance_train_samples.svg#gh-light-mode-only">
  <img align="center" alt="Classes Unbalance" src="/examples/light/classes_unbalance_train_samples.svg#gh-dark-mode-only">
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

It's reasonable to say that some curves are better than the others (the highest and most to the left curves - nearer to the "ROC heaven") however it is very hard to draw any assumptions or conclusions from it. The next option is to calculate the AUC values for every ROC Curve associated with every model.<br>
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

The first observation to be made, and that is valid for all models, is that all AUC values are above 0.90, i.e., even classes with less samples still managed to be decently learned by all models.<br>
Still, the learning of some classes can differ between the models. Take a look at the ゐ class, that has only 400 training samples, the NN_2_adam achieved an AUC value of ~0.97, whereas the NN_2_sgd reached ~0.94; it's a small difference, mas not negligible, even more so when the NN_2_adam had higher accuracy (77.94%) than the NN_2_sgd (70.35%).

#### CNN_1_adam vs CNN_2_sgd
<p align="center">
  <img align="center" alt="CNN_1_adam_auc_x_CNN_2_sgd_auc" src="/examples/dark/results/CNN_1_adam_auc_x_CNN_2_sgd_auc.svg#gh-light-mode-only">
  <img align="center" alt="CNN_1_adam_auc_x_CNN_2_sgd_auc" src="/examples/light/results/CNN_1_adam_auc_x_CNN_2_sgd_auc.svg#gh-dark-mode-only">
</p>

Now comparing the different optimizer's impact on the CNN models. The first thing to be noticed, is that the CNNs consistently present AUC values of at least ~0.97. Next, the AUC values between the models are pretty close, but curiosly enough the ゐ class presents the lowest AUC value for the CNN_2_sgd, in contrast to the stable AUC value of ~0.99 for the CNN_2_adam model. And last, the す class, presents the same dips for both models. 

#### NN_dense vs NN_dense_batch_norm
<p align="center">
  <img align="center" alt="NN_dense_auc_x_NN_dense_batch_norm_auc" src="/examples/dark/results/NN_dense_auc_x_NN_dense_batch_norm_auc.svg#gh-light-mode-only">
  <img align="center" alt="NN_dense_auc_x_NN_dense_batch_norm_auc" src="/examples/light/results/NN_dense_auc_x_NN_dense_batch_norm_auc.svg#gh-dark-mode-only">
</p>

Now it's the dense NN models turn. The only interesting aspect to be pointed out, it's that the AUC values remain between ~0.98 and ~1.00 for all classes. But in fact, is difficult to say anything about the models just by looking at these plots. The final conclusion will only be possible in the next section (spoiler!). 

#### CNN_dense vs CNN_dense_batch_norm
<p align="center">
  <img align="center" alt="CNN_dense_auc_x_CNN_dense_batch_norm_auc" src="/examples/dark/results/CNN_dense_auc_x_CNN_dense_batch_norm_auc.svg#gh-light-mode-only">
  <img align="center" alt="CNN_dense_auc_x_CNN_dense_batch_norm_auc" src="/examples/light/results/CNN_dense_auc_x_CNN_dense_batch_norm_auc.svg#gh-dark-mode-only">
</p>

Surprisingly, for the densest CNNs, it is possible to notice that the batch normalization helped to keep a slightly average AUC value.

#### NN_dense_batch_norm vs CNN_dense_batch_norm
<p align="center">
  <img align="center" alt="NN_dense_batch_norm_auc_x_CNN_dense_batch_norm_auc" src="/examples/dark/results/NN_dense_batch_norm_auc_x_CNN_dense_batch_norm_auc.svg#gh-light-mode-only">
  <img align="center" alt="NN_dense_batch_norm_auc_x_CNN_dense_batch_norm_auc" src="/examples/light/results/NN_dense_batch_norm_auc_x_CNN_dense_batch_norm_auc.svg#gh-dark-mode-only">
</p>

At last, comparing the "batch normalized and densest" NN and CNN models, it's only possible to notice that the CNN kept a higher AUC value, in average, when compared to the NN.<br>
The final results about which is the best NN and CNN model, and which is the best between the two, will be presented in the next section.

---------------------------------------------------------------------
### Weighted Average AUC
From the comparisions made in the last section, it is clear that is difficult to make a clear decision of what is the best model, and in some sentences, an "average AUC value" was mentioned. That is because this is the strategy used to evaluate multiclass AUC analyses.<br>
So, in order to summarize the AUC values associated with every class for every model, a **weighted average** was used, where the classes with less samples have higher weight than the ones with more samples\*. The logic is simple, models that managed to learn classes with less samples, just as good as for classes with more samples, deserves to get a higher score.<br>
The next figure shows the results for the weighted average AUC.

PS.: the weights are calculated as the inverse of the number of samples for each class.

<p align="center">
  <img align="center" alt="auc_weighted_avg" src="/examples/dark/results/auc_weighted_avg.svg#gh-light-mode-only">
  <img align="center" alt="auc_weighted_avg" src="/examples/light/results/auc_weighted_avg.svg#gh-dark-mode-only">
</p>

At last, the final summarization for the AUC values. Conclusions:
- The Adam optimizer granted higher average AUC values, more considerably for the NN models;
- The average AUC value for the CNN models are all very close to each other, meaning it's reasonable comparing them by only using their accuracy;
- As for the NNs, it's better to take both metrics in consideration: accuracy and weighted average AUC;

## Conclusion
To decide which is the best NN and CNN model, and what's the best between the two, the conclusions made from both the accuracy and multiclass AUC analyses will be used. Here is the final plot to help us further:

<p align="center">
  <img align="center" alt="scatter_accuracy_auc" src="/examples/dark/results/scatter_accuracy_auc.svg#gh-light-mode-only">
  <img align="center" alt="scatter_accuracy_auc" src="/examples/light/results/scatter_accuracy_auc.svg#gh-dark-mode-only">
</p>

For the NN models:
- NNs that used the SGD optimizer accomplished the lowest accuracy and AUC values;
- Next the NNs that used the Adam optimizer achieved better accuracy and slightly better AUC values;
- At last, the denser NN models achieved better accuracy and AUC values;

Therefore, the best NN model is the **NN_dense_batch_norm**, with both higher accuracy and AUC values.

As fot the CNN models:
- Similarly to the NN models, the CNNs that used the SGD optimizer got the worst results, with lower accuracy and AUC values;
- However the CNN_dense model got the second lowest accuracy among all CNN models;
- Although the CNN_dense_batch_norm got the best accuracy, followed immediately by the CNN_1_adam and CNN_2_adam models;

Since the CNN_dense_batch_norm, CNN_1_adam and CNN_2_adam all got very similar results, both for the accuracy and multiclass AUC, to decide which was the best, another factor will be taken into consideration: the number of trainable parameters.

|         Model        | # of trainable parameters |
| :------------------ | :-----------------------: |
|       CNN_1_adam     |          201073           |
|       CNN_2_adam     |          922225           |
| CNN_dense_batch_norm |         2888719           |

So, with only 201073 trainable parameters, ~4.6x less than the CNN_2_adam model and ~14.4x less than the CNN_dense_batch_norm, the **CNN_1_adam** achieved pretty much the same performance as more complex models, being also **better than the best NN model**.

## Discussion

TODO - Talk about using ResNets, for reduced computational cost -> less time to train
