import torch 
import torch.nn as nn

class SmoothL1Loss(nn.Module): 
    def __init__(self, beta : float = 1.0, reduction : str = 'mean'): 
        super().__init__() 
        
        self.beta = beta
        self.reduction = reduction
        
    def forward(self, prediction: torch.Tensor, ground_truth: torch.Tensor):
        """
        Calculates the Smooth L1 Loss.

        Args:
            prediction (torch.Tensor): The predicted values, expected to have shape [N, *].
            ground_truth (torch.Tensor): The ground truth values, expected to have shape [N, *].

        Returns:
            torch.Tensor: The calculated Smooth L1 loss.
        """
        diff = prediction - ground_truth
        abs_diff = torch.abs(diff)
        loss = torch.where(abs_diff < self.beta, 0.5 * (diff ** 2) / self.beta, abs_diff - 0.5 * self.beta)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss