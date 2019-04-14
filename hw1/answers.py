r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**  
No.  
In fact increasing k only helps sometimes in the low values, and then quickly makes the accuracy worse.  
That is because when using a value for k that is too large we are underfitting the model.  
That happens because we are looking at too many points in the training set, even those that are too far away.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**  
The choice of delta > 0 is arbitrary for the SVM loss L(W) because delta controls the same trade-off as lambda.  
the bigger the margin between answers needs to be, the bigger the weights can grow in order to meet that expectation  
without changing the actual relations between the scores.  
The same goes for a smaller margin and weights shrinking.  
This means that while delta and lambda are supposedly different hyper-parameters, 
because they both control the same trade-off (weights' size to accuracy) delta can be chosen arbitrarily as long as 
lambda is tuned correctly with said arbitrary value.
"""

part3_q2 = r"""
**Your answer:**  
1) We can interpret what the model is actually learning as trying to find a template for each class.  
The class's template the input is closest to is the one which the model would predict.  
  
2) This is different from KNN because while in KNN we save *every* input and try to find the k nearest ones,  
here, we learn and save a template for each class.  
Basically, instead of remembering every sample, we try to create a point for each class.  
Each class will be best represented by it's point and use it instead.  
In this notion, the KNN with k=1 and the templates being the points it remembers, should be equivalent
to the linear classifier.
"""

part3_q3 = r"""
**Your answer:**  
1) Based on the graph of the training set loss we would say that our learning rate is **good**.  
The value of the average loss converged to small values under 1 after 5-10 epochs and kept getting better afterwards.  
If the learning rate was **too low** the graph would show much slower convergence,  
and if the learning rate was **too high** the loss would probably not converge at all.  
  
2) Based on the graph of the Training and validation sets' accuracy we think that the model is
**slightly overfitted to the training set**.  
We can see that the accuracies of the 2 sets converged into 2 different values,  
and that the test set's value is slightly higher than the validation set's.
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**  
The ideal pattern of a residual plot would be a straight line, exactly on the y-hat axis.  
That plot means that for any given prediction, the correct regression value is equal.  
Based on the above residual plot we can say that the training examples predictions were closer to their
actual score compared to the test examples.  
This could be seen by the fact that the training points are more compact around the y-hat axis.  
The plot gotten by the top-5 features looks much nicer than the final plot as its points are seen to be within about
1 unit of error.
"""

part4_q2 = r"""
**Your answer:**  
1. We use the regularization term in order to get smaller values in the W matrix,  
and in order to prevent overfitting to the training examples.
As explained in the last part changing a bit the value of lambda should'nt affect the learning rate,  
but using a logarithmic space could have more drastic effect on it.

2. There are 3 options for degree and 5 for lambda, hence 15 total possible combinations.  
For any combination we used k-fold with k=3 to fit the data and then test it.
So, in total, the model was fitted 15 * 3 = 45 times to the data.
"""

# ==============
