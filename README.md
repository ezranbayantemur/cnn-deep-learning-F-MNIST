# Clothes Classification on CNN with F-MNIST Dataset
With this project we intended to classify clothing products such as shirts, clothes and shoes by using Deep Learning.

By creating layered artificial neural network with Convolutional Neural Network (CNN), we trained it with fashion_mnist dataset in the Keras library 

As a result our output model can classify a clothes by it's category. 

### Layer Info
Our model uses fashion_mnist dataset as input, which is Keras library about clothes and fashion data. As architectural, our CNN model's created by multiple Conv2D and MaxPooling2D layers with 'relu' activators and multiple Dense layers.

Before view our model, let's get some a short info about layers and their parameters.

*	**Convolutional Layer:** It is layer using for determine the distinguishing features of the input value. Scans the input by the matrix criteria that specified in the parameter.
    *	**_filters_** :_** Filter layer pcs of the model.
    *	**_kernel_size:_** The row - column value of the matrix that will scan the input value.
    *	**_padding:_** It can take two types parameters; "Same" and "Valid". On "Same" value, if the input value is not fully scanned by the CL, the missing fields are assigned as "0". On "Valid" fields that cannot be scanned (less than kernel_size) are clipped automatically.
    *	**_input_shape:_** Size of the input value.
    *	**_activation:_** Function that will handle the reading values   

*	**Pooling (Downsampling) Layer:** Decreases the weight of the value that coming from the Convolutional layer and controls compliance. It also reduces the size.
    * **_pool_size:_** Scanning matrix size of layers to be processed.
    * **_strides:_** The number of units to shift, after the size of the entries is reduced. 

* **Dropout Layer:** Used to protect the model from overfitting (too much learning) 
    * **_rate:_** Determines the percentage of the number of links to be cropped.

* **Dense Layer:** Generally used to create bonds between the neurons by their frequencies
	  * **_units:_** Bond number that will create between neurons

*	**Flatten Layer:** Used to convert values within Fully Connected layer into a single array.

## Our Model
Firstly we have Conv2D layer which is 8 filter, 3x3 kernel size, "Same" padding parameters and "relu" activator.
```python
model.add(Conv2D(filters = 8,
                 kernel_size = (3,3),     
                 padding = 'Same',
                 input_shape = (28, 28, 1),
                 activation = 'relu'))
```
And we create MaxPooling layer for reduce our output value.
```python
model.add(MaxPooling2D(pool_size=(2,2)))
```
After that we cut off 1/4 of neuron connections.
```python
model.add(Dropout(0.25))
```

After these layers we create same layers just diffrence Conv2D filters parameters.
```python
model.add(Conv2D(filters = 16, 
                 kernel_size = (3,3), 
                 padding = 'Same', 
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
```
We created our core layers. Now we flat our values and for display them as image we create our Dense layers with "softmax" activator.
```python
model.add(Flatten())
model.add(Dense(254, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
```

As optimizer we used Adam function for considering it's CNN perfonmance.
```python
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer= optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
```

And lastly we fit our model with 250 batch size and divide it as 150 epoch
```python
model.fit(inp_train,out_train,batch_size=250,verbose=1,epochs=150,validation_split=0.2)
```
As a result we had created our 4 layer CNN model for image classification. Now we can observe it for perfonmance and test values.

