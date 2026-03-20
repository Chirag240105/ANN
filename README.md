# 🧠 Neural Network from Scratch (NumPy) – Explained Line by Line

This project demonstrates a **simple feedforward neural network** using Python and NumPy. Below is a complete explanation of each line of code and why it is used.

---

## 📦 Import Library

```python
import numpy as np
```

* Imports NumPy for numerical operations
* Used for arrays, matrix multiplication, and mathematical functions

---

## 📊 Input and Output Data

```python
x = np.array([[1,0,1,1],[0,0,1,0],[1,1,0,1]])
y = np.array([1,0,1,0])
```

### 🔹 `x` (Input Data)

* Represents input features
* Each row = one sample
* Each column = one feature

Example:

* 3 samples
* 4 features → so input layer = 4 neurons

### 🔹 `y` (Output Data)

* Expected output values (labels)
* Used for training (though training is not implemented here)

---

## ⚙️ Activation Function

```python
def sigmoid_activation(x):
  return 1 / (1 + np.exp(-x))
```

* Sigmoid function converts values into range **(0,1)**
* Used to introduce **non-linearity**
* Helps in making predictions like probabilities

---

## 🔢 Define Network Structure

```python
inputNeurons = x.shape[1]
hiddenNeurons = 4
outputNeurons = 1
```

### 🔹 `inputNeurons`

* Number of input features
* `x.shape[1]` → number of columns → 4

👉 Input layer = number of columns in dataset

### 🔹 `hiddenNeurons`

* Number of neurons in hidden layer (chosen manually)

### 🔹 `outputNeurons`

* Number of outputs (1 for binary classification)

---

## 🎲 Initialize Weights and Biases

```python
weightsHidden = np.random.uniform(size=(inputNeurons, hiddenNeurons))
biasHidden = np.random.uniform(size=(1, hiddenNeurons))
weightOutput = np.random.uniform(size=(hiddenNeurons, outputNeurons))
biasOutput = np.random.uniform(size=(1, outputNeurons))
```

### 🔹 Why Random Initialization?

* Prevents all neurons from learning the same thing
* Helps break symmetry

---

### 🔹 `weightsHidden`

* Shape: `(inputNeurons, hiddenNeurons)`
* Connects input → hidden layer

---

### 🔹 `biasHidden`

* Shape: `(1, hiddenNeurons)`
* One bias per hidden neuron

---

### 🔹 `weightOutput`

* Shape: `(hiddenNeurons, outputNeurons)`
* Connects hidden → output layer

---

### 🔹 `biasOutput`

* Shape: `(1, outputNeurons)`
* One bias per output neuron

---

## 🧠 Why Bias is Important

Mathematical form:

```
f(x) = x · w + bias
```

* Similar to equation of line:

```
y = mx + c
```

* `w` → slope (weights)
* `bias` → shift (like `c`)

👉 Without bias:

* Model always passes through origin (0,0)

👉 With bias:

* Model can shift left/right or up/down
* Makes learning more flexible and powerful

---

## 🔁 Forward Propagation (Step-by-Step)

### 1️⃣ Input → Hidden Layer

```python
fx = np.dot(x, weightsHidden) + biasHidden
```

* Matrix multiplication of input and weights
* Adds bias to shift values

---

### 2️⃣ Apply Activation Function

```python
hiddenLayer = sigmoid_activation(fx)
```

* Applies sigmoid
* Converts raw values into range (0,1)
* Output becomes input for next layer

---

### 3️⃣ Hidden → Output Layer

```python
fx_ = np.dot(hiddenLayer, weightOutput) + biasOutput
```

* Same process again:

  * Multiply by weights
  * Add bias

---

### 4️⃣ Final Output

```python
outputLayer = sigmoid_activation(fx_)
outputLayer
```

* Final prediction
* Values between 0 and 1

---

## 🔄 Complete Flow

```
Input (x)
   ↓
Weights + Bias
   ↓
Hidden Layer (Sigmoid)
   ↓
Weights + Bias
   ↓
Output Layer (Sigmoid)
```

---

## 🚀 Summary

* **Weights** → control importance of connections
* **Bias** → allows shifting and flexibility
* **Sigmoid** → introduces non-linearity
* **Forward propagation** → computes output step-by-step

---

## ⚠️ Note

This code only performs **forward propagation**
It does NOT include:

* Loss calculation
* Backpropagation
* Training loop

---

## 🔥 Next Step

To make this a complete neural network, you can add:

* Loss function (like binary cross-entropy)
* Backpropagation
* Weight updates using gradient descent

---

💡 This is the foundation of deep learning — mastering this means you truly understand how neural networks work under the hood.
