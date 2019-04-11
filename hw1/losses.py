import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        #
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops).
        #   Hint: Create a matrix M where M[i,j] is the margin-loss
        #   for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        # margin loss matrix before rounding up to 0.
        loss_matrix = x_scores - x_scores.gather(1, y.view(-1, 1)) + self.delta
        loss_matrix[range(loss_matrix.shape[0]), y] = 0
        # non negative margin loss matrix.
        loss_matrix.clamp_(0)

        # could'nt find an easy way not to add delta to the correct predictions,
        # so the easiest fix was to just subtract delta from the loss.
        loss = torch.mean(torch.sum(loss_matrix, dim=1))
        # ========================

        # ====== YOUR CODE: ======
        self.grad_ctx['m'] = loss_matrix
        self.grad_ctx['y'] = y
        self.grad_ctx['x'] = x
        # ========================

        return loss

    def grad(self):

        # Same notes as above. Hint: Use the matrix M from above, based on
        # it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        g = torch.zeros_like(self.grad_ctx['m'])
        g[self.grad_ctx['m'] > 0] = 1
        g[range(g.shape[0]), self.grad_ctx['y']] = -torch.sum(g, dim=1)

        grad = torch.mm(self.grad_ctx['x'].t(), g) / self.grad_ctx['x'].shape[0]
        # ========================

        return grad
