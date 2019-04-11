import torch
from torch import Tensor
from torch.utils.data import DataLoader
from collections import namedtuple

from .losses import ClassifierLoss


class LinearClassifier(object):

    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # Initialize it from a normal dist with zero mean and the given std.

        # ====== YOUR CODE: ======
        self.weights = torch.normal(torch.zeros((n_features, n_classes)), weight_std)
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # Calculate the score for each class using the weights and
        # return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = torch.matmul(x, self.weights)
        y_pred = torch.argmax(class_scores, dim=1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # Use the predict function above and compare the predicted class
        # labels to the ground truth labels to obtain the accuracy (in %).
        # Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        acc = (y == y_pred).sum().item() / len(y)
        # ========================

        return acc * 100

    def train(self,
              dl_train: DataLoader,
              dl_valid: DataLoader,
              loss_fn: ClassifierLoss,
              learn_rate=0.1, weight_decay=0.001, max_epochs=100):
        Result = namedtuple('Result', 'accuracy loss')
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print('Training', end='')
        for epoch_idx in range(max_epochs):
            # At each epoch, evaluate the model on the entire training set
            # (batch by batch) and update the weights.
            # Each epoch, also evaluate on the validation set.
            # Accumulate average loss and total accuracy for both sets.
            # The train/valid_res variables should hold the average loss and
            # accuracy per epoch.
            #
            # Don't forget to add a regularization term to the loss, using the
            # weight_decay parameter.

            total_correct = 0
            average_loss = 0

            # ====== YOUR CODE: ======
            # train on the training set while also calculating accuarcy and average loss
            num_iters, total_samples = 0, 0
            for data, target in dl_train:
                classification, class_scores = self.predict(data)

                total_correct += (classification == target).sum().item()
                total_samples += classification.size(0)

                average_loss += loss_fn.loss(data, target, class_scores, classification)
                num_iters += 1

                grad = loss_fn.grad() + weight_decay * self.weights
                self.weights -= learn_rate * grad

            # update accuracy and loss
            train_res.accuracy.append(total_correct * 100.0 / total_samples)
            train_res.loss.append(average_loss * 1.0 / num_iters)

            # calculate accuracy and average loss on the validation set
            total_correct, average_loss = 0, 0
            num_iters, total_samples = 0, 0
            for data, target in dl_valid:
                classification, class_scores = self.predict(data)

                total_correct += (classification == target).sum().item()
                total_samples += classification.size()[0]

                average_loss += loss_fn.loss(data, target, class_scores, classification)
                num_iters += 1

            valid_res.accuracy.append(total_correct * 100.0 / total_samples)
            valid_res.loss.append(average_loss * 1.0 / num_iters)

            # ========================
            print('.', end='')

        print('')
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be at the end).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        w_images = (self.weights[:-1] if has_bias else self.weights).clone()
        w_images = w_images.t().contiguous().view(-1, *img_shape)
        # ========================

        return w_images
