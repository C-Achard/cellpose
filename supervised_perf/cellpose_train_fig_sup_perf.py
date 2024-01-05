from cellpose.models import CellposeModel
from glob import glob
from tifffile import imread
import numpy as np
import pathlib as pt

VAL_PERCENT = 0.01
def convert_2d(images_array, images_names=None, dtype=np.float32):
    images_2d = []
    images_names_2d = [] if images_names is not None else None
    for i, image in enumerate(images_array):
        for j, slice in enumerate(image):
            images_2d.append(slice.astype(dtype))
            if images_names is not None:
                images_names_2d.append(f"{pt.Path(images_names[i]).stem}_{j}.tif")
    return images_2d, images_names_2d

if __name__=="__main__":
    # path_images = pt.Path().home() / "Desktop/Code/CELLSEG_BENCHMARK/TPH2_mesospim/TRAINING"
    path_images = pt.Path("/data/cyril/CELLSEG_BENCHMARK/TPH2_mesospim/TRAINING")
    # path_images = path_images / "SPLITS"
    path_images = path_images / "ALL/"
    X_paths = sorted(glob(str(path_images / '*.tif')))
    Y_paths = sorted(glob(str(path_images / 'labels/*.tif')))

    X = list(map(imread,X_paths))
    Y = list(map(imread,Y_paths))
    X_2d, X_paths_2d = convert_2d(X, X_paths)
    Y_2d, Y_paths_2d = convert_2d(Y, Y_paths, dtype=np.uint16)
    print(len(X_2d))
    print(len(Y_2d))
    
    assert len(X_2d) == len(Y_2d)
    ind = range(len(X_2d))
    
    n_val = max(1, int(round(VAL_PERCENT * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X_2d[i] for i in ind_val]  , [Y_2d[i] for i in ind_val]
    X_trn, Y_trn = [X_2d[i] for i in ind_train], [Y_2d[i] for i in ind_train] 
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))

    X_trn_paths = [X_paths_2d[i] for i in ind_train]
    X_val_paths = [X_paths_2d[i] for i in ind_val]
    print("Train :")
    [print(p) for p in X_trn_paths]
    print("Val :")
    [print(p) for p in X_val_paths]
    print("*"*20)
    
    # import napari
    # viewer = napari.viewer.Viewer()
    # [viewer.add_labels(y) for y in Y_trn[50:70]]
    model = CellposeModel(
        gpu=True,
        pretrained_model=False,
        model_type=None,
        diam_mean=3.3,
        # nchan=1,
    )
    model.train(
        train_data=X_trn,
        train_labels=Y_trn,
        train_files=X_trn_paths,
        test_data=X_val,
        test_labels=Y_val,
        test_files=X_val_paths,
        save_path="./",
        save_every=50,
        n_epochs=50,
        channels=[0,0],
        model_name="supervised_perf_fig.cellpose",
    )
    X_trn[0].shape
