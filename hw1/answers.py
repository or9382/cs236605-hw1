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
The choice of delta > 0 is arbitrary for the SVM loss L(W) because  
delta controls the same trade-off as lambda,  
the bigger the margin between answers needs to be the weights can grow  
in order to meet that expectation without changing the actual relations between the scores,  
and the same goes for a smaller margin and the weights shrinking.  
This means that while delta and lambda are supposedly different hyper-parameters  
because they both control the same trade-off (weights' size to accuracy) delta can be chosen  
arbitrarily as long as lambda is tuned correctly with said arbitrary value.
"""

part3_q2 = r"""
**Your answer:**  
1) We can interpret what the model is actually learning as trying to find a template for each class  
so that if for most inputs the template they are the closest to should be the one for their correct class.   
  
2) This is different from KNN because while in KNN we save *every* input and try to find the k nearest ones,  
here we learn and save a template for each class so that with KNN using k=1 we'll have the best accuracy.  
Basically instead of remembering every sample we try to create a point for each class so that  
each class will be best represented by it's point and save them instead.
"""

part3_q3 = r"""
**Your answer:**  
1) Based on the graph of the training set loss We would say that  
our learning rate is **good** as the value of the average loss converged to  
small values under 1 after 5-10 epochs and kept getting better afterwards.  
If the learning rate was **too low** the graph would show much slower convergence,  
and if the learning rate was **too high** the loss would probably not converge at all.  
  
2) Based on the graph of the Training and validation sets' accuracy we would say that  
the model is **slightly overfitted to the training set** as we can see that  
the accuracies of the 2 sets converge into 2 different values, and the test set's value  
is slightly higher than the validation set's. 
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**  


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**  


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
