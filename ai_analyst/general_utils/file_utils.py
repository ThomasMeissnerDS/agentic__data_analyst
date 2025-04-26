import os
import shutil


def copy_files(
        src: str = "/kaggle/input/dejavusans-bold-ttf/DejaVuSans-Bold.ttf",
        tgt_dir: str = "/kaggle/working/output/dejavusans-bold-ttf",
        tgt_name: str = "DejaVuSans-Bold.ttf",
        ) -> str:
    """
    Copy a file from the input directory to
    the working directory.
    """
    os.makedirs(tgt_dir, exist_ok=True)
    shutil.copyfile(src, os.path.join(tgt_dir, tgt_name))
    return os.path.join(tgt_dir, tgt_name)
