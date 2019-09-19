import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    y=[ x if x == 1 else -1 for x in y ]
    newX=np.insert(X, 0, 1, axis=1)
    #np.append(newX, values=X, axis=1)
    newW=np.insert(w,0, 0, axis=0)
    
    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        fraction=(step_size/N)
        #pred_y=np.dot(newX,newW)
        while max_iterations!=0:
            pred_y=np.dot(newX,newW)
            updatew=np.zeros(D+1)
            for i in range(len(y)):   
                if y[i]*pred_y[i]<=0:
                    updatew+=np.multiply(np.transpose(newX[i]),y[i])
            newW+=(fraction)*updatew
            max_iterations-=1
        ############################################
        w = newW[1:D+1]
        b= newW[0]
    
    elif loss == "logistic":
          
        #w + λ σ(−yw''x)yx
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        
        #y=np.array(y)
        #y=y.reshape(len(y),1)
        fraction=(step_size/N)
        while max_iterations!=0:
            updatew = np.zeros(D+1)
            ''' temp=np.dot(newX,newW)
            temp=temp.reshape(len(temp),1)
            z_array=sigmoid(-1*y*temp)
            productyx= np.dot(np.transpose(y*newX), z_array)
            newW+=fraction*productyx'''
            """if y[i]==0:
                correct_y=-1
            else:
                correct_y=1"""
            z= sigmoid(-1*np.multiply(y,(np.dot(newX,np.transpose(newW)))))
            #z=(1/N)*(step_size)*sigmoid(-1*z)
            #w+=z*correct_y*X[i]
            product=np.multiply(z,y)
            updatew+=np.dot(newX.T,product)
            #w+=1/N*(step_size)*sigmoid(-1*np.dot(np.dot(y, np.transpose(w)), X))*y*X
            #b+=(z*correct_y)
            newW+=fraction*updatew
            max_iterations-=1
        ############################################
  
        w = newW[1:D+1]
        b= newW[0]

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = np.divide(1,np.add(1,np.exp(-1*z)))
    ############################################
    
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    preds = list()
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        array=np.dot(X,w)+b
        for i in range(len(X)):
            prdicted_val=array[i]
            if prdicted_val>0:
                preds.append(1.)
            else:
                preds.append(0.)
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        array=np.dot(X,w)+b
        apply_sigmoid=sigmoid(array)
        for current_pred in apply_sigmoid:
            if current_pred>0.5:
                preds.append(1.)
            else:
                preds.append(0.)
                
    else:
        raise "Loss Function is undefined."
    
    preds=np.asarray(preds)
    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros((C,1))
    if b0 is not None:
        b = b0
    #X=np.insert(X, 0, 1, axis=1)
    np.random.seed(42)
    #w = np.zeros((C, D+1))
    if gd_type == "sgd":
        while max_iterations!=0:
            current_choice=np.random.choice(N)
            shuffled_x=X[current_choice]
            ############################################
            # TODO 6 : Edit this if part               #
            #          Compute w and b                 #
            current_class= np.zeros((C,1))
            yn=y[current_choice]
            current_class[yn]=1
            #print(b.shape,w.shape)
            temp_w=np.dot(shuffled_x,np.transpose(w))+b.T#-np.dot(shuffled_x,np.transpose(w[yn]))+b.T-b[yn]
            #print('temp',temp)
            z_w=softmax_gd(temp_w.T)
            gw=z_w-current_class
            #gw=gw.reshape(len(gw),1)
            #print(w.shape, gw.shape, shuffled_x.shape)
            shuffled_x=shuffled_x.reshape(1,len(shuffled_x))
            #print(w.shape, gw.shape, shuffled_x.shape)
            w=w-(step_size*np.dot(gw, shuffled_x))
            b=b-(step_size*gw)
            #print('final',w)
            max_iterations-=1
        
        #b= w[:,0].T
        #w =w[:,1:D+2]
        b=b.reshape(len(w),)
        #b=np.array(b)
        print(w.shape, b.shape, C, D)
        

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D+1))
        X=np.insert(X, 0, 1, axis=1)
        one_hot_matrix=np.zeros((N,C))
        for i,c in enumerate(y):
            one_hot_matrix[i][c]=1 
            
        while max_iterations!=0:
            temp=np.dot(w,X.T)-np.sum(np.multiply(np.dot(one_hot_matrix,w),X),axis=1).T
            z=softmax_gd(temp)-one_hot_matrix.T
            w=w-((step_size/N)*(np.dot(z, X)))       
        ############################################
            max_iterations-=1

        b= w[:,0].T
        w =w[:,1:D+2]  
        ############################################
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    newX=np.insert(X, 0, 1, axis=1)
    #np.append(newX, values=X, axis=1)
    newW=np.insert(w,0, b, axis=1)
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)
    ############################################
    probs = softmax(np.dot(newX,np.transpose(newW)))
    preds = np.argmax(probs,axis=1)
    assert preds.shape == (N,)
    return preds

def softmax(z):
    #print(z.shape)
    z -= np.max(z)
    sm = np.exp(z) / np.sum(np.exp(z))
    return sm

def softmax_gd(z):
    #print(z.shape)
    z -= np.max(z,axis=0)
    sm = np.exp(z) / np.sum(np.exp(z),axis=0)
    return sm


        