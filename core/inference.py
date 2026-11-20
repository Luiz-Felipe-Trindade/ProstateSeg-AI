from pathlib import Path
import numpy as np
import torch
import cv2
import SimpleITK as sitk

def predict_volume_masks(
    model: torch.nn.Module,
    volume: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
    image_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    Generate binary segmentation masks for all axial slices of a 3D MRI volume.

    Parameters
    ----------
    model
        Trained segmentation model.

    volume
        Input MRI volume with shape (depth, height, width).

    device
        PyTorch device used for inference.

    threshold
        Probability threshold used to obtain binary predictions.

    image_size
        In-plane size used by the segmentation model.

    Returns
    -------
    np.ndarray
        Binary segmentation volume with the same shape as the input volume.
    """
    model.eval()

    original_height = volume.shape[1]
    original_width = volume.shape[2]

    predicted_masks = []

    with torch.no_grad():
        for slice_index in range(volume.shape[0]):
            image_slice = (
                volume[slice_index]
                .astype(np.float32)
            )

            non_zero_mask = (
                image_slice != 0
            )

            normalized_slice = (
                image_slice.copy()
            )

            if non_zero_mask.any():
                values = normalized_slice[
                    non_zero_mask
                ]

                mean = values.mean()
                std = values.std()

                if std > 0:
                    normalized_slice[
                        non_zero_mask
                    ] = (
                        values - mean
                    ) / std

            resized_slice = cv2.resize(
                normalized_slice,
                image_size,
                interpolation=cv2.INTER_LINEAR,
            )

            image_tensor = (
                torch.from_numpy(
                    resized_slice
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            logits = model(
                image_tensor
            )

            probabilities = torch.sigmoid(
                logits
            )

            prediction = (
                probabilities
                >= threshold
            )

            predicted_mask = (
                prediction[0, 0]
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            restored_mask = cv2.resize(
                predicted_mask,
                (
                    original_width,
                    original_height,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

            restored_mask = (
                restored_mask > 0
            ).astype(np.uint8)

            predicted_masks.append(
                restored_mask
            )

    return np.stack(
        predicted_masks,
        axis=0,
    )

def save_predicted_mask(
    predicted_volume: np.ndarray,
    reference_image,
    output_path: str | Path,
) -> None:
    """
    Save a predicted binary segmentation volume while preserving
    the spatial metadata of the reference MRI image.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predicted_image = sitk.GetImageFromArray(
        predicted_volume.astype(np.uint8)
    )

    predicted_image.CopyInformation(
        reference_image
    )

    sitk.WriteImage(
        predicted_image,
        str(output_path),
    )

