import numpy as np
from scipy.ndimage import rotate, zoom, shift
from typing import Optional

class VolumeTransformer:
    """
    A class for storing and transforming 3D numpy arrays with support for rotation along Z,
    scaling with interpolation (separate XY and Z scales), and center point translation.
    """

    def __init__(self, array: np.ndarray, xy_scale: float = 1.0, z_scale: float = 1.0):
        """
        Initialize the transformer with a 3D numpy array and scaling parameters.

        Args:
            array (np.ndarray): 3D numpy array (e.g., medical volume).
            xy_scale (float): Current XY scaling factor.
            z_scale (float): Current Z scaling factor.
        """
        if array.ndim != 3:
            raise ValueError("Array must be 3D")
        self.array = array.copy()
        self.xy_scale = xy_scale
        self.z_scale = z_scale

    def rotate_z(self, angle_degrees: float, order: int = 3):
        """
        Rotate the array around the Z-axis.

        Args:
            angle_degrees (float): Rotation angle in degrees.
            order (int): Interpolation order (0-5, default 3 for cubic).

        Returns:
            VolumeTransformer: New instance with rotated array.
        """
        # Array shape (Z, Y, X), rotate in XY plane, axes=(1, 2)
        rotated = rotate(self.array, angle_degrees, axes=(1, 2), order=order, mode='constant', cval=0)
        return VolumeTransformer(rotated, self.xy_scale, self.z_scale)

    def scale(self, xy_scale_factor: float, z_scale_factor: float, order: int = 3):
        """
        Scale the array with separate factors for XY (horizontal) and Z (vertical) axes.

        Args:
            xy_scale_factor (float): Scaling factor for XY axes.
            z_scale_factor (float): Scaling factor for Z axis.
            order (int): Interpolation order (0-5, default 3 for cubic).

        Returns:
            VolumeTransformer: New instance with scaled array and updated scales.
        """
        # Array shape is (Z, Y, X), so zoom_factors = (z_factor, xy_factor, xy_factor)
        zoom_factors = (z_scale_factor, xy_scale_factor, xy_scale_factor)
        scaled = zoom(self.array, zoom_factors, order=order, mode='constant', cval=0)
        new_xy_scale = self.xy_scale * xy_scale_factor
        new_z_scale = self.z_scale * z_scale_factor
        return VolumeTransformer(scaled, new_xy_scale, new_z_scale)

    def translate(self, dx: float, dy: float, dz: float, order: int = 3):
        """
        Translate the array by shifting along X, Y, Z axes (moves the center point).

        Args:
            dx (float): Shift in X direction.
            dy (float): Shift in Y direction.
            dz (float): Shift in Z direction.
            order (int): Interpolation order (0-5, default 3 for cubic).

        Returns:
            VolumeTransformer: New instance with translated array.
        """
        shift_vector = (dx, dy, dz)
        translated = shift(self.array, shift_vector, order=order, mode='constant', cval=0)
        return VolumeTransformer(translated, self.xy_scale, self.z_scale)

    def crop(self, desired_shape: tuple):
        """
        Crop the array to the desired shape, retaining the center point.

        Args:
            desired_shape (tuple): Desired shape (depth, height, width). Use None to keep current size for that axis.

        Returns:
            VolumeTransformer: New instance with cropped array.
        """
        current_shape = self.array.shape
        slices = []
        for i, (curr, des) in enumerate(zip(current_shape, desired_shape)):
            if des is None or des >= curr:
                slices.append(slice(None))
            else:
                start = (curr - des) // 2
                end = start + des
                slices.append(slice(start, end))
        cropped = self.array[tuple(slices)]
        return VolumeTransformer(cropped, self.xy_scale, self.z_scale)
    
    def normal(self, min: float = 0, max: float = 1):
        s_min = self.array.min()
        s_max = self.array.max()
        normalised = (self.array - s_min) / (s_max - s_min)
        return VolumeTransformer(normalised, self.xy_scale, self.z_scale)

    def get_array(self) -> np.ndarray:
        """Get the current array."""
        return self.array

    def get_scales(self) -> tuple:
        """Get current XY scale and Z scale."""
        return self.xy_scale, self.z_scale

'''   
class VolumeCT:
    def __init__(self, array: np.ndarray, xy_spasing: float = 1.0, z_spasing: float = 1.0):
        """
        Initialize the container with a 3D numpy array and scale parameters.

        Args:
            array (np.ndarray): 3D numpy array (CT volume).
            xy_spasing (float): Current XY voxel size.
            z_spasing (float): Current Z voxel size.
        """
        if array.ndim != 3:
            raise ValueError("Array must be 3D")
        self.array = array.copy()
        self.xy_spasing = xy_spasing
        self.z_spasing = z_spasing
    
    def center_crop(self, s_x: Optional[float] = None,  s_y: Optional[float] = None,  s_z: Optional[float] = None):
        c_z, c_y, c_x = c_shape = self.array.shape()
        b_z, b_y, b_x = [(c_i-s_i)//2 for c_i, s_i in zip(c_shape, sizes)] # border
        volume = self.array[b_z : b_z+c_z, b_y : b_y+c_y, b_x : b_x+c_x]


    def rotate_z(self, angle_degrees: float, order: int = 3, crop=False):
        """
        Rotate the array around the Z-axis.

        Args:
            angle_degrees (float): Rotation angle in degrees.
            order (int): Interpolation order (0-5, default 3 for cubic).

        Returns:
            VolumeTransformer: New instance with rotated array.
        """
        # Array shape (Z, Y, X), rotate in XY plane, axes=(1, 2)
        rotated = rotate(self.array, angle_degrees, axes=(1, 2), order=order, mode='constant', cval=0)
        volume = VolumeTransformer(rotated, self.xy_scale, self.z_scale)
        if crop:
            '''