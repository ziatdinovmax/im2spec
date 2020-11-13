from typing import Tuple, Type, Union

import numpy as np
import torch

from shapely.geometry import Polygon


def predict(model, feature_arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Makes a prediction with a trained NN
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(feature_arr, np.ndarray):
        feature_arr = torch.from_numpy(feature_arr).float()
    feature_arr = feature_arr.to(device)
    with torch.no_grad():
        prediction = model(feature_arr)
    prediction = prediction.cpu().numpy()
    return prediction


def init_dataloaders(X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_test: np.ndarray,
                     y_test: np.ndarray,
                     batch_size: int
                     ) -> Tuple[Type[torch.utils.data.DataLoader]]:
    """
    Returns train and test dataloaders for training images
    in a native PyTorch format
    """
    device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_train = torch.from_numpy(X_train).float().to(device_)
    y_train = torch.from_numpy(y_train).float().to(device_)
    X_test = torch.from_numpy(X_test).float().to(device_)
    y_test = torch.from_numpy(y_test).float().to(device_)

    data_train = torch.utils.data.TensorDataset(X_train, y_train)
    data_test = torch.utils.data.TensorDataset(X_test, y_test)
    train_iterator = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True)
    test_iterator = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size)
    return train_iterator, test_iterator


def make_window(imgsrc: np.ndarray, window_size: int,
                xpos: int, ypos: int) -> np.ndarray:
    """
    Returns the portion of the image within the window given the
    image (imgsrc), the x position and the y position
    """
    imgsrc = imgsrc[int(xpos-window_size/2):int(xpos+window_size/2), 
                    int(ypos-window_size/2):int(ypos+window_size/2)]
    return imgsrc


def create_training_set(hdata: np.ndarray, window_size: int,
                        slice: int = 0) -> Tuple[np.ndarray]:
    """
    Creates arrays with features (local subimages)
    and targets (corresponding spectra) from hyperspectral data
    """
    feature_arr, target_arr = [], []
    pos = []
    s1, s2 = hdata.shape[:-1]
    for i in range(s1):
        for j in range(s2):
            arr_loc = make_window(hdata[..., slice], window_size, i, j)
            if arr_loc.shape != (window_size, window_size):
                continue
            feature_arr.append(arr_loc)
            target_arr.append(hdata[i, j, :])
            pos.append([i, j])
    return (np.array(pos), np.array(feature_arr)[:, None],
            np.array(target_arr)[:, None])


def loop_area(pred_loop: np.ndarray,
              target_arr: np.ndarray,
              spec_val: np.ndarray) -> Tuple[np.ndarray]:
    """
    Calculates loop area for predicted and ground truth data
    and computes absolute error
    """
    polygons, polygons_pred = [], []
    for val1, val2 in zip(target_arr[:, 0], pred_loop[:, 0]):
        polygon, polygon_pred = [], []
        for i, v in enumerate(spec_val):
            polygon.append([v, val1[i]])
            polygon_pred.append([v, val2[i]])
        polygons.append(polygon)
        polygons_pred.append(polygon_pred)
    pred_area = np.array([Polygon(p).area for p in polygons_pred])
    target_area = np.array([Polygon(p).area for p in polygons])
    area_error = np.abs(pred_area-target_area)
    return pred_area, target_area, area_error
