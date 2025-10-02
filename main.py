import MONAI
import reporter

import flask


def monai_infer(volume):
        print(f'monai_infer got object of type {type(volume)}')
        try:
            print(f'as volume of shape {volume.array.shape}')
            
            p = MONAI.main(volume)
            print(f"{p = } indeed")
            info = {
                'probability_of_pathology': p,
                'pathology': 1 if p > 0.7 else 0
            }
            return info
        except Exception as e:
            print("The volume couldn't be processed by MONAI")
            raise ValueError(f"Error on CT volume processing: {e}")

# Фласка нет, а надо
if __name__ == "__main__":
    import tempfile, zipfile
    ds_path = r"C:\Users\goroh\Downloads\Датасет.zip"

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(ds_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        
        xlsx_file = reporter.process(temp_dir, monai_infer)