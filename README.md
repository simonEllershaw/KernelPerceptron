# Kernel Perceptron

This repo contains a numpy implementation of the kernel perceptron for the MNIST Digit classification task. The performance of the algorithm is compared empircally to that of the vanilla perceptron, SVM and CNN algorithms on the MNIST digit classifcation task.

## Kernel Perceptron Algorithm

This alogrithm is an extension of the basic kernel perceptron. However instead of a dot product between the input and support vectors a kernel operation is used. This allows the modelling of non-lineraties. The algorithm is summarised neatly by this [wikipedia post](https://en.wikipedia.org/wiki/Kernel_perceptron).

To extend the perceptron to a multi-class problem "one vs all" approach has been used. 10 perceptron classes are trained, one for each digit, to discriminate between class digit and not class digit. Inference is then performed by each perceptron returing the signed distance of the input to their hyperplane. The input is then assigned to the class for which this is the maxiumum. 

The calculation of the kernel is the most expensive step of the algoithm, attempts to increase the model efficeny have been made in the following ways
+   Only the vectors which have been used to correct mistakes (i.e. alpha ≠ 0) are saved as support vectors. This reduces the size of the inference matrix
+   At training time the kernel matrix is symetric so calculated for the top diagonal then transposed
+   Kernel matrices shared between percetrons are only calculated once

## Comparison Algotrithms

The kernel perceptron model was compared to 3 others:
1.  Vanilla perceptron, using the dot product as the kernel
2.  SVM, using [Ski-kit's SVM](https://scikit-learn.org/stable/modules/svm.html) implementation. Little additional work was required as the library is fairly self contained
3. A Pytorch implementation of the simple [LeNet5](https://en.wikipedia.org/wiki/LeNet) CNN architecture. A small amount of modification was required at the flattening stage due to the smaller 16x16 input size of the MNIST digit dataset.

## Results

The results shown in the table below are an average over 5 runs of each algorothim. It is of note that an extenisive hyperparamter search has not been undertaken for any of the models.

| Model                             | Training Time\s | Inference Time\s | Accuracy\\%  |
|-----------------------------------|-----------------|------------------|-------------|
| Kernel Perceptron                 | 114 ± 2   | 11.5 ± 0.2     | 0.963 ± 0.002 |
| Vanilla Perceptron                | 95 ± 2    | 17.3 ± 0.2     | 0.894 ± 0.005 |
| SVM (3rd Order Polynomial Kernel) | 10.8 ± 0.1    | 2.118 ± 0.009      | 0.961 ± 0.001 |
| CNN- LeNet5                       | 24 ± 2    | 0.25 ± 0.01      | 0.961 ± 0.003 |
