
import torch 


def convert_parameters_for_training(taskT:float, freq_policy:float) -> list[int, int, int]:
    
    samples     = torch.floor(torch.tensor(taskT*freq_policy, dtype=torch.int32)).item()
    
    return samples