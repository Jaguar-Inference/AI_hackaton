import pydicom
import os
import numpy as np
from PIL import Image
import tempfile, zipfile
from volume_utils import VolumeTransformer

def load_dicom_series(file_paths):
    slices = []
    
    for file_path in file_paths:
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=False)
            if ds.Modality != 'CT':
                continue
            slices.append(ds)
        except Exception as e:
            pass
    assert slices, "DICOM CTs not found"

    info = {
        'study_uid': str(slices[0].get('StudyInstanceUID', None)),
        'series_uid': str(slices[0].get('SeriesInstanceUID', None))
    }

    # Сортировка по Instance Number или Slice Location
    def slice_sort_key(ds):
        if 'InstanceNumber' in ds:
            return int(ds.InstanceNumber)
        elif 'SliceLocation' in ds:
            return float(ds.SliceLocation)
        else:
            return 0

    slices = sorted(slices, key=slice_sort_key)
    

    # Сборка 3D массива из отдельных срезов
    pixel_arrays = [s.pixel_array for s in slices]
    volume = np.stack(pixel_arrays, axis=0)
    volume_norm = (volume - volume.min()) / (volume.max() - volume.min())

    try:
        xy_scale = slices[0].PixelSpacing
    except:
        xy_scale = 1.0
    z_positions = [ds.ImagePositionPatient[2] for ds in slices] if slices and hasattr(slices[0], 'ImagePositionPatient') else []
    z_scale = np.abs(np.diff(z_positions)).mean() if len(z_positions) > 1 else 1.0
    # info = {'xy_spacing': xy_spacing, 'z_scale': z_scale}
    
    return VolumeTransformer(volume_norm.astype(np.float32), xy_scale, z_scale), info

def make_grid(images, size=64):
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    gridsize = int(np.ceil(len(images) ** 0.5))
    output_im = Image.new("RGB", [gridsize * size,]*2)
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), ((i % gridsize) * size, (i // gridsize) * size))
    return output_im

def load_dicom_from_zip(zip_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Распаковываем ZIP архив
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            except Exception as e:
                raise ValueError("the .zip file is broken")
            # Находим все DICOM файлы
            dicom_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    dicom_files.append(os.path.join(root, file))

            if not dicom_files:
                raise ValueError("В архиве не найдены DICOM файлы")

            print(f"Найдено {len(dicom_files)} DICOM файлов")
            # Загружаем все DICOM файлы
            volume, info = load_dicom_series(dicom_files)
            return volume, info
        except Exception as e:
            raise ValueError(f"Can't load dicoms from zip: {e}")
        

if __name__ == "__main__":
    """
    # Пример использования — путь к папке с файлами
    folder_path = 'data/norma_anon'
    filepaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

    volume, info = load_dicom_series(filepaths)

    print('XY spacing:', info['xy_spacing'], 'Z scale:', info['z_scale'])
    print('Original scales:', volume.get_scales())
    print('Volume shape:', volume.get_array().shape)

    # Transform to target spacing for model input (e.g., 1.0, 1.0, 15)
    target_xy_spacing = 1.0
    target_z_spacing = 15
    current_xy, current_z = volume.get_scales()
    xy_factor = current_xy / target_xy_spacing
    z_factor = current_z / target_z_spacing
    volume = volume.scale(xy_factor, z_factor, order=2)  # Use linear interpolation for speed
    volume = volume.rotate_z(45, order=2)

    print('Resampled scales:', volume.get_scales())
    print('Resampled shape:', volume.get_array().shape)"""
    #volume, info = load_dicom_from_zip("data/norma_anon.zip"); print(info)
    volume = VolumeTransformer(np.load("C:/Users/goroh/Downloads/scan_0006.npy"))
    c_shape = volume.array.shape
    c_height, c_width = c_shape[0], min(c_shape[1:])
    volume = volume.normal().crop((c_height,c_width,c_width)) # normalise, convert base to square
    volume = volume.scale(z_scale_factor= 40/c_height, xy_scale_factor= 128/c_width, order=1)

    arr = volume.get_array()
    print(arr.shape)
    arr = arr.transpose(0, 2, 1)



    slicelist = [Image.fromarray((slice_arr * 255).astype(np.uint8))
                 for slice_arr in arr]
    image = make_grid(slicelist, arr.shape[1]) # volume.array.shape[1])
    image.show()
