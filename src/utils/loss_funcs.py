
import torch.nn as nn
import torch


class DepthUncertaintyLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gaussian_nll = nn.GaussianNLLLoss(eps=eps)

    def forward(self, depth_pred, var_pred, depth_gt, logvar=False):
        var_pred = nn.functional.relu(var_pred) # TODO: find a better way to ensure positive variance
        if logvar:
            var_pred = torch.exp(var_pred).clamp(min=self.eps)
        return self.gaussian_nll(depth_pred, depth_gt, var_pred) 


def scale_invariant_rmse(predicted, ground_truth):
    """
    predicted: Tensor of shape (B, 1, H, W)
    ground_truth: Tensor of shape (B, 1, H, W)
    Returns: scalar tensor (loss value)
    """

    # Flatten spatial dimensions
    B = predicted.size(0)
    predicted = predicted.reshape(B, -1)
    ground_truth = ground_truth.reshape(B, -1)

    # Log difference
    log_diff = torch.log(predicted) - torch.log(ground_truth)

    # Compute the global bias (alpha)
    alpha = torch.mean(log_diff, dim=1, keepdim=True)
    # Subtract bias and compute RMSE
    corrected_diff = log_diff - alpha  # Important! Subtract bias before squaring
    loss = torch.sqrt(torch.mean(corrected_diff ** 2, dim=1))

    return loss.mean() # scalar


def mae_loss(predicted, ground_truth):
    """
    Mean Absolute Error (MAE)
    predicted: Tensor of shape (B, 1, H, W)
    ground_truth: Tensor of shape (B, 1, H, W)
    Returns: scalar tensor
    """
    return torch.mean(torch.abs(predicted - ground_truth))


def rmse_loss(predicted, ground_truth):
    """
    Root Mean Squared Error (RMSE)
    """
    return torch.sqrt(torch.mean((predicted - ground_truth) ** 2))


def rel_loss(predicted, ground_truth, epsilon=1e-6):
    """
    Mean Relative Error (REL)
    Avoids division by zero using epsilon.
    """
    return torch.mean(torch.abs(predicted - ground_truth) / (ground_truth + epsilon))


def delta_accuracy(predicted, ground_truth, threshold):
    """
    Computes the percentage of pixels where max(pred/gt, gt/pred) < threshold
    """
    epsilon = 1e-6
    max_ratio = torch.max(
        predicted / (ground_truth + epsilon),
        ground_truth / (predicted + epsilon)
    )
    return torch.mean((max_ratio < threshold).float())


def delta1_accuracy(predicted, ground_truth):
    return delta_accuracy(predicted, ground_truth, 1.25)


def delta2_accuracy(predicted, ground_truth):
    return delta_accuracy(predicted, ground_truth, 1.25 ** 2)


def delta3_accuracy(predicted, ground_truth):
    return delta_accuracy(predicted, ground_truth, 1.25 ** 3)