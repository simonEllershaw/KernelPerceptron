from Models.multiClassKernelPerceptron import MultiClassKernelPerceptron
import Models.cnn as cnn
from Models.svm import SVM
import Models.kernelFunctions as kernelFunctions
from Data.mnistDigitLoader import MnistDigits
from Data.mnistDigitLoaderPyTorch import MnistDigitsPytorch

import time
import numpy as np
from statistics import mean, stdev

def calc_accuracy(predict, ground_truth):
    return np.count_nonzero(predict==ground_truth[:len(predict)]) / float(len(predict))

def metrics_from_dict_lists(data_dict):
    average_dict = {}
    sterr_dict = {}
    for key, values in data_dict.items():
        average_dict[key] = mean(values)
        # As defined by Hughes and Hayes
        sterr_dict[key] = stdev(values) / len(values)**0.5
    return average_dict, sterr_dict

def table_results(time_train, time_predict, accuracy):
    time_train_mean, time_train_sterr = metrics_from_dict_lists(time_train)
    time_predict_mean, time_predict_sterr = metrics_from_dict_lists(time_predict)
    accuracy_mean, accuracy_sterr = metrics_from_dict_lists(accuracy)

    print("|Model|Train Time +- Err|Predict Time +- Err|Accuracy +- Err|")
    for model_name in time_train.keys():
        print(f"|{model_name} | {time_train_mean[model_name]:.3f} {time_train_sterr[model_name]:.3f} | {time_predict_mean[model_name]:.3f} {time_predict_sterr[model_name]:.3f} | {accuracy_mean[model_name]:.3f} {accuracy_sterr[model_name]:.3f}|")

def main(data_fname, number_iterations=10):
    # Init metrics
    time_train = {"kernel_perceptron":[],
        "vanilla_perceptron":[],
        "svm":[],
        "cnn":[],
    }

    time_predict = {"kernel_perceptron":[],
        "vanilla_perceptron":[],
        "svm":[],
        "cnn":[],
    }

    accuracy = {"kernel_perceptron":[],
        "vanilla_perceptron":[],
        "svm":[],
        "cnn":[],
    }

    for iteration in range(number_iterations):
        print(f"--------Iteration {iteration}-----------")
        # Init models
        vanilla_perceptron= MultiClassKernelPerceptron(kernelFunctions.dot_product)
        kernel_perceptron = MultiClassKernelPerceptron(kernelFunctions.polynomial_kernel, 3)
        svm = SVM(3, 'poly', hyperparameters=3)
        leNet5 = cnn.LeNet5()

        # Load data splits
        data = MnistDigits(data_fname).get_split_datasets()
        dataloaders = MnistDigitsPytorch.getDataLoader(data, batch_size=12)

        # Train models
        t1 = time.time()
        print("Train Kernel Perceptron")
        kernel_perceptron.train(data["images_train"], data["labels_train"], data["images_val"], data["labels_val"])
        t2 = time.time()
        print("Train Vanilla Perceptron")
        vanilla_perceptron.train(data["images_train"], data["labels_train"], data["images_val"], data["labels_val"])
        t3 = time.time()
        print("Train SVM")
        svm.train(data["images_train"], data["labels_train"])
        t4 = time.time()
        print("Train CNN")
        cnn.train(leNet5, dataloaders)
        t5 = time.time()
        time_train["kernel_perceptron"].append(t2-t1)
        time_train["vanilla_perceptron"].append(t3-t2)
        time_train["svm"].append(t4-t3)
        time_train["cnn"].append(t5-t4)

        # Predict with trained models
        t1 = time.time()
        print("Predict Kernel Perceptron")
        y_pred_kernel_perceptron = kernel_perceptron.predict(data["images_test"])
        t2 = time.time()
        print("Predict Vanilla Perceptron")
        y_pred_vanilla_perceptron = vanilla_perceptron.predict(data["images_test"])
        t3 = time.time()
        print("Predict SVM")
        y_pred_svm = svm.predict(data["images_test"])
        t4 = time.time()
        print("Predict CNN")
        y_pred_cnn = cnn.predict(leNet5, dataloaders["test"])
        t5 = time.time()
        time_predict["kernel_perceptron"].append(t2-t1)
        time_predict["vanilla_perceptron"].append(t3-t2)
        time_predict["svm"].append(t4-t3)
        time_predict["cnn"].append(t5-t4)

        # Calc accuracy
        accuracy["kernel_perceptron"].append(calc_accuracy(y_pred_kernel_perceptron, data["labels_test"]))
        accuracy["vanilla_perceptron"].append(calc_accuracy(y_pred_vanilla_perceptron, data["labels_test"]))
        accuracy["svm"].append(calc_accuracy(y_pred_svm, data["labels_test"]))
        accuracy["cnn"].append(calc_accuracy(y_pred_cnn, data["labels_test"]))

    table_results(time_train, time_predict, accuracy)

main(r"Data\large.dat")