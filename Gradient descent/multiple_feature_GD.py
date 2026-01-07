
import numpy as np
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

def load_data():
    """
    Loads the diabetes dataset from sklearn.
    """
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    return X, y

def z_score_normalize_features(X):
    """
    Normalizes the features using Z-score normalization.
    X_norm = (X - mu) / sigma
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    mu = np.mean(X, axis=0) # mean of each feature
    sigma = np.std(X, axis=0) # std of each feature
    
    # avoid division by zero if any feature has 0 std (constant feature)
    sigma[sigma == 0] = 1
    
    X_norm = (X - mu) / sigma
    
    return X_norm, mu, sigma

def compute_cost(X, y, w, b):
    """
    Computes the cost function for linear regression.
    
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): The cost J(w,b) 
    """
    m = X.shape[0]
    cost = 0.0
    f_wb = np.dot(X, w) + b
    cost = np.sum((f_wb - y)**2) / (2 * m)
    return cost

def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    err = (np.dot(X, w) + b) - y
    dj_db = np.sum(err) / m
    dj_dw = np.dot(X.T, err) / m
        
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha.
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      J_history (list) : History of cost values
    """
    J_history = []
    w = w_in
    b = b_in
    
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)   
        
        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Save cost J at each iteration
        if i < 100000:      # prevent resource exhaustion 
            J_history.append(cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % (num_iters // 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}")
        
    return w, b, J_history

def main():
    # 1. Load Data
    X_original, y = load_data()
    
    # 2. Normalize Features
    X_norm, mu, sigma = z_score_normalize_features(X_original)
    
    print(f"Original shape: {X_original.shape}")
    print(f"Normalized shape: {X_norm.shape}")
    print(f"Mean of features: {mu[:4]} ...") # print just a few
    print(f"Std of features: {sigma[:4]} ...") # print just a few

    m, n = X_norm.shape
    
    # 3. Initialize Parameters
    w_init = np.zeros((n,))
    b_init = 0.
    
    # 4. Settings for Gradient Descent
    iterations = 1000
    alpha = 0.1 # Learning rate
    
    print("\nStarting Gradient Descent...")
    
    # 5. Run Gradient Descent
    w_final, b_final, J_hist = gradient_descent(X_norm, y, w_init, b_init, 
                                                compute_cost, compute_gradient, 
                                                alpha, iterations)
    
    print(f"\nFinal cost: {J_hist[-1]:0.2f}")
    print(f"Final weights (w): {w_final}")
    print(f"Final bias (b): {b_final:0.2f}")
    
    # Optional: Plotting (if environment allows, otherwise just print is fine)
    # plt.plot(J_hist)
    # plt.xlabel("Iteration")
    # plt.ylabel("Cost")
    # plt.title("Cost vs Iteration")
    # plt.show()

if __name__ == "__main__":
    main()
