from tensorflow.keras import Input, Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
from tensorflow.keras.utils import to_categorical
from matplotlib.pyplot import figure, imshow, ion, legend, plot, savefig, show, subplots, title, xlabel, ylabel
from numpy import argmax, asarray, dstack, mean, sqrt, unique
from os import environ
from pandas import read_csv
from seaborn import histplot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from warnings import filterwarnings
filterwarnings("ignore")
environ["KMP_DUPLICATE_LIB_OK"], environ["TF_CPP_MIN_LOG_LEVEL"], environ["TF_ENABLE_ONEDNN_OPTS"] = "TRUE", "3", "0"

# A class for plotting evaluation of model's performance on test set epoch by epoch
class TestCallback(Callback):
    def __init__(self):
        self.loss, self.acc = [], []

    def on_epoch_end(self, epoch, logs = {}):
        Loss, accuracy = self.model.evaluate(X_test, caty_test, verbose = 0)
        self.loss.append(Loss)
        self.acc.append(accuracy)
        # print(f"Testing loss: {Loss}, Accuracy: {accuracy}")

# Loading & normalizing the dataset
trainSet, testSet, inputLayer, numberFilters, filterSize, ReLU, Softmax, poolSize, Strides, lossFunction, Accuracy, Loss, validationAccuracy, validationLoss, Epochs, Title, Alpha, localPopulation, Numbers, numberIterations, numberGenerations, Fitness, Parameters, Optimizer, Legends, Location, XLabel, YLabel, Blue, Green, Red, figureSize, DPI, train_batch_size, test_batch_size, validationSplit, OFF, Axis, Tuner = read_csv("mnist_train.csv"), read_csv("mnist_test.csv"), Input(shape = (48, 48, 3)), [2, 4, 6, 8, 10], (3, 3), "relu", "softmax", (2, 2), 2, "categorical_crossentropy", "accuracy", "loss", "val_accuracy", "val_loss", 15, "Convolutional Neural Network Evaluation", 0.6, 1, 10, 1, 1, [], [], "adam", ["Train Loss", "Validation Loss", "Test Loss"], "upper right", "Epochs", "Loss", "blue", "green", "red", (13, 5), 1200, 1000, 100, 0.2, 0, 1, {}
df, Length, rangeEpochs, Metric, XAcc = trainSet._append(testSet, ignore_index = True), int(sqrt(trainSet.shape[1])), range(Epochs), [Accuracy], Accuracy.title()
X_train, X_test, y_train, y_test = trainSet.drop(["label"], axis = 1).astype("float32") / 255., testSet.drop(["label"], axis = 1).astype("float32") / 255., trainSet["label"].astype("int"), testSet["label"].astype("int")
dfX_train, localBound, number_train_samples, number_test_samples, numberClasses  = X_train, X_train.shape[1], len(X_train) // 2, len(X_test), len(unique(y_test))

# Halving train set due to complexity of VGG16 model!
subsequentLayers, X_train, y_train = [BatchNormalization(), MaxPooling2D(pool_size = poolSize, strides = Strides), Flatten(), Dense(8, activation = ReLU), Dense(4, activation = ReLU), Dense(numberClasses, activation = Softmax)], X_train.iloc[:number_train_samples], y_train.iloc[:number_train_samples]
X_train, X_test, caty_train, caty_test = X_train.to_numpy().reshape(-1, 28, 28, 1), X_test.to_numpy().reshape(-1, 28, 28, 1), to_categorical(y_train, numberClasses), to_categorical(y_test, numberClasses)
KFold, inputShape = StratifiedKFold(n_splits = 5, shuffle = True).split(X_train, y_train), X_train.shape[1:]
priorLayers, TCB = [Input(shape = inputShape), Conv2D(6, kernel_size = filterSize, activation = ReLU), BatchNormalization(), MaxPooling2D(pool_size = poolSize, strides = Strides)], TestCallback()
print(f"There're {number_train_samples} samples of {Length} X {Length} images as train samples and {number_test_samples} samples as test samples.")

"""
# Plot samples
for Number in range(Numbers):
    fig = figure
    imshow(arrX_train[Number], cmap = "gray")
    savefig(dpi = 1200)
    show()

# Plotting raw data records
Figure, ax = subplots()
ax.pie(df["label"].value_counts(), labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], autopct = '%1.1f%%', shadow = True, startangle = 90)
ax.axis("equal")
savefig(dpi = 1200)
show()
histplot(df["label"])
savefig("MNIST Histogram", dpi = 1200)
show()
"""

print("\nTuning classifier...")
for Filters in numberFilters:
    Accuracies = []
    for Train, Validation in KFold:
        # Create model
        Classifier = Sequential(priorLayers + [Conv2D(Filters, kernel_size = filterSize, activation = ReLU)] + subsequentLayers)

        Classifier.compile(loss = lossFunction,  metrics = Metric, optimizer = Optimizer)

        # Fit the model
        Classifier.fit(X_train[Train], caty_train[Train], batch_size = 128, epochs = 3, verbose = OFF)

        # Evaluate the model
        Accuracies.append(Classifier.evaluate(X_train[Validation], caty_train[Validation], batch_size = 64, verbose = OFF)[1] * 100)
    Tuner[Filters] = mean(Accuracies)
    # Store the mean score and the filter number
    
print("\n\nTraining the classifier...\n")
# Defining the model
CNNClassifier = Sequential(priorLayers + [Conv2D(max(Tuner, key = Tuner.get), kernel_size = filterSize, activation = ReLU)] + subsequentLayers)

# Compiling the model
CNNClassifier.compile(loss = lossFunction, metrics = Metric, optimizer = Optimizer)
print(CNNClassifier.summary())
CNNHistory = CNNClassifier.fit(X_train, caty_train, batch_size = train_batch_size, callbacks = [TCB], epochs = Epochs, validation_split = validationSplit, verbose = OFF)

# Saving model
CNNClassifier.save("CNN Classifier.keras")

# Plotting the training history (losses)
figure(figsize = figureSize)
ion()
plot(CNNHistory.history[Loss], Blue)
plot(CNNHistory.history[validationLoss], Green)
plot(rangeEpochs, TCB.loss, Red)
title(Title)
ylabel(YLabel)
xlabel(XLabel)
legend(Legends, loc = Location)
savefig(Title, dpi = DPI)
show()

# Plot the training history (accuracies)
figure(figsize = figureSize)
ion()
plot(CNNHistory.history[Accuracy], Blue)
plot(CNNHistory.history[validationAccuracy], Green)
plot(rangeEpochs, TCB.acc, Red)
title(Title)
ylabel(XAcc)
xlabel(XLabel)
legend(Legends, loc = Location)
savefig(Title, dpi = DPI)
show()

Title = "Convolutional Neural Network Confusion Matrix"
print(f"\nEvaluating classifier...\n\nModel's score: {CNNClassifier.evaluate(X_test, caty_test, batch_size = test_batch_size, verbose = OFF)}")
ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, argmax(CNNClassifier.predict(X_test, batch_size = test_batch_size, verbose = OFF), axis = Axis))).plot()
title(Title)
savefig(Title, dpi = DPI)
show()

# Data preprocessing for being input for VGG16 model
X_train, X_test, TCB, Title = dstack([X_train] * 3).reshape(-1, 28, 28, 3), dstack([X_test] * 3).reshape (-1, 28, 28, 3), TestCallback(), "VGG16 Neural Network Evaluation"
X_train, X_test = asarray([img_to_array(array_to_img(Image, scale = False).resize((48, 48))) for Image in X_train]), asarray([img_to_array(array_to_img(Image, scale = False).resize((48, 48))) for Image in X_test])

# Loading pretrained VGG16 model
VGG16Model = VGG16(input_tensor = inputLayer, include_top = False, weights = None)
VGG16Model.load_weights("VGG16 Pretrained Models/VGG16 Weights(without Top).h5")
VGG16Layers = VGG16Model.layers

# Freezing all the layers of VGG16 model to avoid them from being trained again!
for VGG16Layer in VGG16Layers:
    VGG16Layer.trainable = False

print(VGG16Model.summary())

# Creating model with input and output layer by means of VGG16 pretrained model
VGG16Classifier = Model(inputs = inputLayer, outputs = Dense(numberClasses, activation = Softmax)(Dense(4, activation = ReLU)(Dense(8, activation = ReLU)(Flatten()(VGG16Model.output)))))

print(VGG16Classifier.summary())

VGG16Classifier.compile(loss = lossFunction, metrics = Metric, optimizer = Optimizer)

# Train the model
VGG16History = VGG16Classifier.fit(X_train, caty_train, batch_size = train_batch_size, callbacks = [TCB], epochs = Epochs, validation_split = validationSplit, verbose = OFF)

# Plotting the training history (losses)
figure(figsize = figureSize)
ion()
plot(VGG16History.history[Loss], Blue)
plot(VGG16History.history[validationLoss], Green)
plot(rangeEpochs, TCB.loss, Red)
title(Title)
ylabel(YLabel)
xlabel(XLabel)
legend(Legends, loc = Location)
savefig(Title, dpi = DPI)
show()

# Plot the training history (accuracies)
figure(figsize = figureSize)
ion()
plot(VGG16History.history[Accuracy], Blue)
plot(VGG16History.history[validationAccuracy], Green)
plot(rangeEpochs, TCB.acc, Red)
title(Title)
ylabel(XAcc)
xlabel(XLabel)
legend(Legends, loc = Location)
savefig(Title, dpi = DPI)
show()

Title = "VGG16 Confusion Matrix"
print(f"\nEvaluating VGG16 classifier...\n\nModel's score: {VGG16Classifier.evaluate(X_test, caty_test, batch_size = test_batch_size, verbose = OFF)}")
ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, argmax(VGG16Classifier.predict(X_test, batch_size = test_batch_size, verbose = OFF), axis = Axis))).plot()
title(Title)
savefig(Title, dpi = DPI)
show()
