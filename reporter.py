import pandas as pd
import tempfile
from pathlib import Path
import time

import loader

DEFAULTS = {
    "path_to_study": "",
    "study_uid": "",
    "series_uid": "",
    "probability_of_pathology": float("NaN"),
    "pathology": 0,
    "processing_status": "",
    "time_of_processing": 0
}

def save_report(studys):
    defaults = {k: "" for s in studys for k in s.keys()} | DEFAULTS # find all column names
    #studys = [defaults | s.items for s in studys] # ensure all columns populated
    data = {k: [d[k] if k in d else defaults[k] for d in studys] for k in defaults.keys()} # dict[str, list] <- list[dict]
    temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    pd.DataFrame(data).to_excel(temp_file.name, index=False)
    print(f"Report saved to: {temp_file.name}")
    return temp_file

def process(dir, f):
    report = []
    temp_path = Path(dir)
    for zip_path in temp_path.rglob('*.zip'):
        starttime = time.perf_counter()
        state = {"path_to_study": zip_path.relative_to(temp_path)}
        try:
            volume, info = loader.load_dicom_from_zip(zip_path)
            state.update(info)
            state.update(f(volume))
        except Exception as e:
            state["processing_status"] = "Failure"
            state["exception"] = str(e)
        state["time_of_processing"] = int(time.perf_counter() - starttime)
        report.append(state)
    
    return save_report(report)

if __name__ == "__main__":
    from random import randint
    import zipfile
    def magic_f(volume):
        print(f'magic_f got {type(volume)} as volume')

        magic = randint(0,110)
        print(f"{magic = } indeed")
        if magic > 100: raise ValueError("Magic error")
        info = {
            'probability_of_pathology': magic * 0.01,
            'pathology': 1 if magic > 70 else 0
        }
        return info
    
    ds_path = r"C:\Users\goroh\Downloads\Датасет.zip"
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(ds_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        print(process(temp_dir, magic_f).name)