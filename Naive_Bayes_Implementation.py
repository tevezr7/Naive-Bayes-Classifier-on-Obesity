import numpy as np

class NaiveBayes: #Naive Bayes classifier
    
    def fit(self, x, y): # Fit the model to the training data
        n_samples, n_features = x.shape  # Number of samples and features 
        self._classes = np.unique(y)  # Unique classes in the target variable
        n_classes = len(self._classes)  # Number of unique classes

        self._mean = np.zeros((n_classes, n_features))  # Mean of each feature for each class
        self._var = np.zeros((n_classes, n_features))  # Variance of each feature for each class
        self._prior = np.zeros(n_classes) # Prior probabilities of each class

        for idx, c in enumerate(self._classes):  #compute mean, variance and prior for each class ,c0 = not obese, c1 = obese
            x_c = x[y == c]
            self._mean[idx, :] = x_c.mean(axis=0)  # Mean of features for class c
            self._var[idx, :] = x_c.var(axis=0)
            self._prior[idx] = x_c.shape[0] / n_samples  # Prior probability of class c

    
    def predict(self, x):  # Predict the class labels for the input samples
        y_pred = [self._predict(x_i) for x_i in x] # For each sample, predict the class label
        return np.array(y_pred) # Return the predicted class labels

    def _predict(self, x_i):  # Predict the class label for a single sample
        posteriors = [] # List to store posterior probabilities for each class

        for idx, c in enumerate(self._classes): ## For each class, compute the posterior probability
            prior = np.log(self._prior[idx]) #get log prior | log(prior) = log(p(c))
            posterior = np.sum(np.log(self._pdf(idx, x_i))) #calculate gaussian probability density function for each feature and take log to avoid underflow | log(p(x|c))
            #log posterior = sum the log of the pdf for each feature | log(p(x|c)) = sum(log(p(x1|c)), log(p(x2|c)), ...)
            posteriors.append(prior + posterior) #total posterior = log(prior) + log(posterior)  , posteriors = scorenotobese, scoreobese 
        
        return self._classes[np.argmax(posteriors)]  # Return the class with the highest posterior probability 

    def _pdf(self, class_idx, x):  # Probability density function for Gaussian distribution
        mean = self._mean[class_idx] # Mean of the features for class class_idx
        var = self._var[class_idx] # Variance of the features for class class_idx
        # Gaussian probability density function formula: p(x|c) = (1 / sqrt(2 * pi * var)) * exp(-(x - mean)^2 / (2 * var))
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator 

