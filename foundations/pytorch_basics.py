import torch
import torch.nn

class Solution:
    def reshape(self, to_reshape: torch.Tensor) -> torch.Tensor:
        return torch.round(torch.reshape(to_reshape, (((to_reshape.shape[0] * to_reshape.shape[1]) // 2), 2)), decimals=4)

    def average(self, to_avg: torch.Tensor) -> torch.Tensor:
        return torch.round(torch.mean(to_avg, dim=0), decimals=4)
        

    def concatenate(self, cat_one: torch.Tensor, cat_two: torch.Tensor) -> torch.Tensor:
        return torch.round(torch.cat((cat_one, cat_two), dim=1), decimals=4)

    def get_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.round(torch.nn.functional.mse_loss(prediction, target), decimals=4)
