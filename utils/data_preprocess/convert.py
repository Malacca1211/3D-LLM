"""
Create solids by the DeepCAD dataset
"""
import json
import argparse
import os
import logging
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from converter import DeepCADReconverter
import signal
from contextlib import contextmanager

# 设置日志
log_dir = Path("/home/malacca/3DLLM/logs/data_preprocess")
if not log_dir.exists():
    log_dir.mkdir(parents=True)
logging.basicConfig(
    filename=log_dir / "json2cad_convert_errors.log",
    level=logging.ERROR,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Time out function
@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)
    try:
        yield
    except TimeoutError:
        raise Exception("time out")
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
def raise_timeout(signum, frame):
    raise TimeoutError

NUM_TRHEADS = 36
NUM_FOLDERS = 100
DATA_FOLDER = "/home/malacca/3DLLM/data/cad_json"  # 在这里直接指定数据文件夹路径
OUTPUT_FOLDER = "/home/malacca/3DLLM/data/cad_data"  # 在这里直接指定输出文件夹路径
VERBOSE = True  # 是否打印额外的转换失败信息

def load_json_data(pathname):
    """Load data from a json file"""
    with open(pathname, encoding='utf8') as data_file:
        return json.load(data_file)


def convert_folder_parallel(data):
    fileName, output_folder = data
    save_fileFolder = Path(output_folder) / fileName.stem
    if not save_fileFolder.exists():
        save_fileFolder.mkdir()

    data = load_json_data(fileName)
    reconverter = DeepCADReconverter(data, fileName)

    try:
        with timeout(30):
            reconverter.parse_json(save_fileFolder)
    except Exception as ex:
        error_msg = f"文件 {fileName} 转换失败: {str(ex)[:50]}"
        logger.error(error_msg)
        return [fileName, str(ex)[:50]]
    return None

def find_file_id(file_or_save_folder):
    """
    The json file will have a pathname of the form

    /foo/bar/0040/00408502.json

    The save folder will have a pathname of the form

    /foo2/bar2/0040/00408502/

    We want to find what DeepCAD call the file id.  In this 
    example it's '0040/00408502'
    """
    parent = file_or_save_folder.parent
    return f"{parent.stem}/{file_or_save_folder.stem}"


def find_files_already_processed_in_sub_folder(sub_folder):
    already_processed_ids = set()
    for save_folder in sub_folder.glob("*"):
        file_id = find_file_id(save_folder)
        already_processed_ids.add(file_id)
    return already_processed_ids


def find_files_already_processed_in_output_folder(output_folder):
    already_processed_ids = set()
    for sub_folder in output_folder.glob("*"):
        already_processed_ids = set.union(
            already_processed_ids,
            find_files_already_processed_in_sub_folder(sub_folder)
        )
    return already_processed_ids


if __name__ == "__main__":
    output_folder = Path(OUTPUT_FOLDER)
    if not output_folder.exists():
        output_folder.mkdir()

    # Find the list of files which were already 
    # processed
    already_processed_ids = find_files_already_processed_in_output_folder(output_folder)
    
    # Pre-load all json data
    deepcad_json = []
    skexgen_obj = []
    data_folder = Path(DATA_FOLDER)
    
    for i in range(NUM_FOLDERS): 
        cur_in = data_folder / str(i).zfill(4)
        cur_out = output_folder / str(i).zfill(4)
        if not cur_out.exists():
            cur_out.mkdir()
        files = []
        for f in cur_in.glob("**/*.json"):
            file_id = find_file_id(f)
            if not file_id in already_processed_ids:
                files.append(f)
        deepcad_json += files
        skexgen_obj += [cur_out]*len(files)

    num_files_still_to_process = len(deepcad_json)
        
    assert len(skexgen_obj) == num_files_still_to_process, "JSON & OBJ length different"

    print(f"Found {len(deepcad_json)} files which require processing")
    
    if num_files_still_to_process > 0:
        # Parallel convert to JSON & STL
        iter_data = zip(
            deepcad_json,
            skexgen_obj,
        )
    
        convert_iter = Pool(NUM_TRHEADS).imap(convert_folder_parallel, iter_data) 
        for invalid in tqdm(convert_iter, total=len(deepcad_json)):
            if invalid is not None:
                if VERBOSE:
                    print(f'Error converting {invalid}...')
