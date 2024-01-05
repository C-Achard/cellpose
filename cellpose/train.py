import os
from pathlib import Path


if __name__=="__main__":
    train_path = (Path().home() / "Desktop/cellpose/stacks").absolute()
    os.system(f"python -m cellpose --train --dir {str(train_path)} --n_epochs 100 --pretrained_model None --diam_mean=3.3 --verbose --use_gpu --do_3D --chan 0 --chan2 0")