import mnistDigitLoader
import numpy as np

data = mnistDigitLoader.MnistDigits(r"Data\dtest123.dat")

num_epochs = 2
w = np.zeros((data.get_image_size()**2))
m = np.zeros((num_epochs))


for i in range(num_epochs):
    for t in range(len(data)):
        # Get sample
        x_t = data.get_image(t)
        y = data.get_label(t)
        # Make prediction
        y_hat = -1 if w.T@x_t < 0 else 1

        if y_hat != y:
            print(y_hat, y)
            w += y*x_t
            m[i] += 1

print(m)
print(w)
    
