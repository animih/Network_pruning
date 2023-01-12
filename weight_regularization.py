import torch

def l1_norm(list_of_parameters, alpha = 1):
    reg_cost = 0
    for parameters in list_of_parameters:
        reg_cost += alpha * torch.abs(parameters).sum() / torch.numel(parameters)

    return reg_cost

def l1_s_norm(list_of_parameters, alpha = 1, beta = 1):
    reg_cost = 0
    for parameters in list_of_parameters:
        reg_cost += alpha * (
            beta * (torch.abs(parameters) / (1 + torch.abs(parameters))).sum()
            / torch.numel(parameters)
            + (1-beta) * torch.abs(parameters).sum() / torch.numel(parameters)
            )

    return reg_cost

def l2_norm(list_of_parameters, alpha = 1):
    reg_cost = 0
    for parameters in list_of_parameters:
        reg_cost += alpha * (
            (parameters**2).sum() / torch.numel(parameters)
            )

    return reg_cost

def l2_super_norm(list_of_parameters, alpha = 1, beta = 1):
    reg_cost = 0
    for parameters in list_of_parameters:
        reg_cost += alpha * (
            beta * beta * ( parameters**2 / (1 + parameters**2)).sum()
            / torch.numel(parameters)
            (1-beta) * (parameters**2).sum() / torch.numel(parameters)
            )

    return reg_cost