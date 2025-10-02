import torch
import torch.nn as nn
import numpy as np
from monai.networks.nets import DenseNet121
from PIL import Image

# 1. Определение модели
class LungCTClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super(LungCTClassifier, self).__init__()
        self.base_model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        x = self.sigmoid(x)
        return x

def prepare_input(normalized_array):
    """
    Args:
        normalized_array: np.array формы (Height, Width, Depth)

    Returns:
        torch.Tensor: тензор формы (1, 1, Height, Width, Depth)
    """
    input_tensor = torch.from_numpy(normalized_array).float()
    input_tensor = input_tensor.unsqueeze(0) 
    input_tensor = input_tensor.unsqueeze(0) 

    # Нормализуем в диапазон [0, 1]
    if input_tensor.max() > 1.0:
        input_tensor = input_tensor / input_tensor.max()

    return input_tensor


def predict(normalized_array, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Args:
        normalized_array: np.array формы (Height, Width, Depth)
        model: обученная модель для классификации
        device: устройство для вычислений ('cuda' или 'cpu')
    Returns:
        prediction: 0 (норма) или 1 (патология)
        probability: вероятность класса "патология"
    """
    input_batch = prepare_input(normalized_array)
    input_batch = input_batch.to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_batch)
        probability = output.cpu().numpy()[0][0]
        prediction = 1 if probability > 0.5 else 0

    return prediction, probability

import loader
import os

def main(volume):
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Используется устройство: {device}")

        model = LungCTClassifier().to(device)
        model.load_state_dict(torch.load('best_lung_ct_model.pth'))
    except Exception as e:
        raise Exception(f"Can't initialise the model: {e}")
    
    try:
        c_shape = volume.array.shape
        c_height, c_width = c_shape[0], min(c_shape[1:])
        volume = volume.normal().crop((c_height, c_width, c_width))  # normalise, convert base to square
        volume = volume.scale(z_scale_factor=40 / c_height, xy_scale_factor=128 / c_width, order=1)
        arr = volume.get_array()
        arr = arr.transpose(2, 1, 0)

    except Exception as e:
        raise ValueError("CT volume is not of right shape and can't be recovered")

    try:
        prediction, probability = predict(arr, model, device)
        return probability
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        raise ValueError(f"CT inference pass error: {e}")


if __name__ == "__main__":

    main()
