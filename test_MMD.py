import torch
from datetime import datetime


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x_tile = x.view(x_size, 1, dim)
    x_tile = x_tile.repeat(1, y_size, 1)
    y_tile = y.view(1, y_size, dim)
    y_tile = y_tile.repeat(x_size, 1, 1)
    return torch.exp(-torch.mean((x_tile - y_tile) ** 2, dim=2) / float(dim))


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)


if __name__ == "__main__":
    torch.manual_seed(123)
    batch = 1000
    x = torch.randn(batch, 1)
    y_baseline = torch.randn(batch, 1)
    y_pred = torch.zeros(batch, 1)
    
    print("MMD baseline", compute_mmd(x, y_baseline))
    print("MMD prediction", compute_mmd(x, y_pred))

