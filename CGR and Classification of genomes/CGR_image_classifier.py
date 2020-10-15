
# coding: utf-8

# In[23]:


import scipy
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import os
import imageio
import cv2


# In[49]:

def load_training_data(DIR, CAT):    
    #Parameters:
    #  DIR: directory of training data
    #  CAT: list of categories, corresponding to folders in DIR of training data used for labelling, e.g. ['Candida', 'Yeast']
    #Returns:
    #  training_data: 1-D array of CGR image (values for input layer), shape = 351624, number of training images
    #  y: label of training data corresponding to index in CAT, shape = number of training images, 1
    
    
    training_data = np.array([])
    y = np.array([])
    for i in CAT:
        classifcation = CAT.index(i)
        path = os.path.join(DIR, i)
        for img in os.listdir( path):

            if str(img) == '.DS_Store':
                continue

            image = np.array( imageio.imread( os.path.join(path,img), as_gray = True))
            image = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 



            image = image.flatten()/255


            if not training_data.any():
                training_data = image.reshape(len(image),1)
            else:    
                training_data = np.append(training_data, image.reshape(len(image),1), axis = 1)  
            y = np.append(y, classifcation)


    y = y.reshape(len(y), 1)
    
    #Shuffle training dataset
    rng = np.random.default_rng()
    training_data = np.append(training_data, y.T, axis = 0)
    rng.shuffle(training_data, axis = 1)
    y = training_data[training_data.shape[0] - 1]
    training_data = np.delete(training_data,training_data.shape[0] - 1, axis = 0 )

    y = y.reshape(len(y), 1)
    
    return training_data, y


# In[8]:

def initialize_parameters(layer_dims):
    #Initialize weights and bias for NN
    #Parameters:
    #  layer_dims: list of number of nodes in each layer. layer_dims[0] = input layer = training_data.shape[0]
    #Returns: 
    #  parameters: dict containing weights and bias arrays for each layer. W shape = (layer_dims[l], layer_dims[l - 1])
    L = len(layer_dims)
    parameters = {}
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[(l-1)])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        
    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W,A) + b
    cache = (A, W, b)
    
    return Z, cache

def sigmoid(z):
    cache = z
    A = (1 / (1 + np.exp(-z)))
    return A, cache


def relu(z):
    A = np.maximum(0,z)
    cache = z
    return A, cache
    

def linear_activation_forward(A_prev, W, b, activation):
    
    #Calculates activation function for a layer
    #Parameters:
    # A_prev: Activation from previous layer or input, 
    # W: weight matrix from parameters dict for layer
    # B: bias vector from parameters dict for layer
    # activation: activation function type for layer, sigmoid or relu
    #Returns:
    # A: product of activation function
    # cache: stored values of A and Z for backprop
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = [linear_cache, activation_cache]

    return A, cache

def L_model_forward(X, parameters, keep_prob = 0.86):
    """
    Single pass of forward propigation in neural net
    Parameters:
        X: input vector
        parameters: dict of weights and bias for each layer
        keep_prob: probability of retaining a neuron during dropout regularization
    Returns:
        AL: output of neural network. vector of outputs for all training examples
        cashes: stored values of dropped neurons, activation and Z for each layer
    
    """

    caches = []
    A = X
    L = len(parameters) // 2    # number of layers in the neural network
    
    for l in range(1, L):
        #dropout regularization
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)] , 'relu')
        drop = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
        A = np.multiply(A, drop) #cancelling out droppd units
        A /= keep_prob #inverted dropout
        cache.append(drop) # cache dropped neurons for back prop
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)] , 'sigmoid')
    caches.append(cache)
    
    assert(AL.shape == (1, X.shape[1]))
            
    return AL, caches

def compute_cost(AL, Y):
    """
    Calculates cost for output vector compared to expected valye
    Parameters:
        AL: output vector for each training example from neural netowrk
        Y: labels of training examples
    Returns:
        cost: deviation between output and expected
    """
    
    m = Y.shape[1]
    #cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = -(1 / m) * np.sum(Y*np.log(AL) + (1 - Y)*np.log(1 - AL))

    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost


# In[9]:

def linear_backward(dZ, cache):
    """
    Calculates backprop for single linear layer

    Parameters:
        dZ: derivative of the cost with respect to the linear output (of current layer l)
        cache: tuple of values (A_prev, W, b) coming from the forward propagation in the current layer, retrieves stored values

    Returns:
        dA_prev: Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW: Gradient of the cost with respect to W (current layer l), same shape as W
        db: Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    
    dW = (1/m)*np.dot(dZ,A_prev.T) 
    db = (1/m)*np.sum(dZ,axis =1 , keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    
    

    
    return dA_prev, dW, db


# In[53]:

def relu_backward(dA, cache):
    """
    calculates backprop for a single RELU unit.

    Parameters:
        dA: derivative of post-activation , of any shape
        cache: 'Z' where we store for computing backward propagation efficiently

    Returns:
        dZ:Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


# In[10]:

def sigmoid_backward(dA, cache):
    """
    calculate the backprop for a single SIGMOID unit.

    Parameters:
        dA: derivative of post-activation , of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
        dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


# In[11]:

def linear_activation_backward(dA, cache, activation):
    """
    calculate backprop for single layer
    
    Parameters:
        dA: post-activation derivative for current layer l 
        cache: tuple of values (linear_cache, activation_cache) we stored for layer l during forward prop
        activation: the activation function to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
        dA_prev: Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW: Gradient of the cost with respect to W (current layer l), same shape as W
        db: Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


# In[12]:

def L_model_backward(AL, Y, caches, keep_prob = 0.86):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
        AL: probability vector (shape 1, #examples) , output of the forward propagation (L_model_forward())
        Y: true "label" vector (containing 0 if CAT[0], 1 if CAT[1])
        caches: list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
        grads: A dictionary with the gradients
                 grads["dA" + str(l)] = ... 
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation derivate of loss 
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[(L-1)]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = 'sigmoid')
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        dA = grads["dA" + str(l+1)]
        current_cache = caches[l]
        
        drop_cache = current_cache.pop( len( current_cache) - 1)
        dA = np.multiply(dA, drop_cache)
        dA /= keep_prob
        
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA, current_cache, activation = 'relu')
       
        
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# In[14]:


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Parameters:
        parameters: python dictionary containing your parameters 
        grads: python dictionary containing your gradients, output of L_model_backward
    
    Returns:
        parameters:  python dictionary containing your updated parameters 
                      parameters["W" + str(l)] = ... 
                      parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
    return parameters


# In[15]:

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Parameters:
        X: data, numpy array of shape (number of examples, num_px * num_px * 3)
        Y: true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims: list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate: learning rate of the gradient descent update rule
        num_iterations: number of iterations of the optimization loop
        print_cost: if True, it prints the cost every 100 steps
    
    Returns:
        parameters: parameters learnt by the model. They can then be used to predict.
    """
    drop = np.array([])
    costs = []
    # keep track of cost
    
    # Parameters initialization. 
    parameters = initialize_parameters(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
        if cost < 0.25: # break early to prevent overfitting
            return parameters
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# In[ ]:

# Dimensions of Neural Network to train
#layers_dims = [training_data.shape[0], 50, 20, 15, 1]

# Train neural net on training data, will print cost every 100 iterations
#temp_parameters = L_layer_model(training_data, y.T, layers_dims, learning_rate= .025, num_iterations = 2500, print_cost = True)


# In[58]:

# If successful reduction in cost save parameters as numpy array file
#np.save('2020-10-14 nn human yeast parameters', temp_parameters)


# In[ ]:

def predict_single_yeast_candida(image_path, parameters, cat):
    """
    This function is used to predict species on single image.
    
    Parameters:
        image_path: file location of CGR genome image
        parameters: parameters of the trained model
        cat: list of labels where index specifices prediction value, e.g. output of 0 corresponds to candida classification 
    
    Returns:
        p: predictions for the given dataset X
    """
    image = np.array( imageio.imread(image_path, as_gray = True))

    plt.imshow(image, cmap = 'gray')
    plt.show()
    image = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 

        

    image = image.flatten()/255
    X = image.reshape(len(image),1)
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            print('It is: ' + str(cat[1]))
            p[0,i] = 1
        else:
            print('It is: ' + str(cat[0]))
            p[0,i] = 0
    

        
    return p


# In[16]:

def predict_accuracy(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Parameters:
        X:data set of examples you would like to label
        parameters:parameters of the trained model
    
    Returns:
        p:predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p


# In[57]:

#p = predict_accuracy(training_data, y.T, temp_parameters)


# In[68]:

DIR = '/Users/willcheney/CGR Test/'
CAT = ['Human','Yeast']

def load_test_data(DIR, CAT):   

    test_data = np.array([])
    test_y = np.array([])
    for i in CAT:
        classifcation = CAT.index(i)
        path = os.path.join(DIR, i)
        for img in os.listdir( path):
            if str(img) == '.DS_Store':
                continue
            
            image = np.array( imageio.imread( os.path.join(path,img), as_gray = True))
            image = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 



            image = image.flatten()/255
            if not test_data.any():
                test_data = image.reshape(len(image),1)
            else:    
                test_data = np.append(test_data, image.reshape(len(image),1), axis = 1)  
            test_y = np.append(test_y, classifcation)
    
    test_y = test_y.reshape(len(test_y), 1)

            
    #shuffle       
    rng = np.random.default_rng()
    test_data = np.append(test_data, test_y.T, axis = 0)
    rng.shuffle(test_data, axis = 1)
    test_y = test_data[test_data.shape[0] - 1]
    test_data = np.delete(test_data,test_data.shape[0] - 1, axis = 0 )

    test_y = test_y.reshape(len(test_y), 1)
    
    return test_data, test_y



# In[69]:




# In[72]:




# In[67]:




# In[47]:




# In[80]:




# In[81]:




# In[ ]:



