import torch 
import torch.nn as nn

class SmoothL1Loss(nn.Module): 
    def __init__(self, alpha : float = 0.5, beta : float = 0.5): 
        super().__init__() 
        
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, prediction : torch.Tensor, ground_truth : torch.Tensor):
        """
        Calculates the Smooth L1 Loss.

        Args:
            prediction (torch.Tensor): The predicted positions of the 17 key points, expected to have shape [N, 17]
            ground_truth (torch.Tensor): The ground truth positions of the 17 key points, expected to have shape [N, 17]

        Returns:
            The Smooth L1 loss.
        """
        l1_loss = self.l1_loss(prediction, ground_truth)
        l2_loss = self.l2_loss(prediction, ground_truth)
        
        return self.alpha * l1_loss + self.beta * l2_loss