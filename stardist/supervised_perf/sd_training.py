from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
# matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist import Rays_GoldenSpiral
from stardist.matching import matching, matching_dataset
from stardist.models import Config3D, StarDist3D, StarDistData3D

np.random.seed(42)
lbl_cmap = random_label_cmap()
# Data
import pathlib as pt
VAL_PERCENT = 1e-10
SAVE_PATH = pt.Path("/data/cyril") / "CELLSEG_BENCHMARK/cellseg3d_train" / "SUP_PERF_FIG"

if __name__ == "__main__":
    path_images = pt.Path("/data/cyril/CELLSEG_BENCHMARK/TPH2_mesospim/TRAINING") / "ALL/"
    X = sorted(glob(str(path_images / '*.tif')))
    Y = sorted(glob(str(path_images / 'labels/*.tif')))
    # assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))
    X = list(map(imread,X))
    Y = list(map(imread,Y))
    n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]

    axis_norm = (0,1,2)   # normalize channels independently
    # axis_norm = (0,1,2,3) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 3 in axis_norm else 'independently'))
        sys.stdout.flush()

    X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]

    assert len(X) > 1, "not enough training data"
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(VAL_PERCENT * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))

    def plot_img_label(img, lbl, img_title="image (XY slice)", lbl_title="label (XY slice)", z=None, **kwargs):
        if z is None:
            z = img.shape[0] // 2    
        fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
        im = ai.imshow(img[z], cmap='gray', clim=(0,1))
        ai.set_title(img_title)    
        fig.colorbar(im, ax=ai)
        al.imshow(lbl[z], cmap=lbl_cmap)
        al.set_title(lbl_title)
        plt.tight_layout()
    i = 0
    img, lbl = X[i], Y[i]
    assert img.ndim in (3,4)
    img = img if img.ndim==3 else img[...,:3]
    # plot_img_label(img,lbl)
    # Configuration
    print(Config3D.__doc__)
    extents = calculate_extents(Y)
    anisotropy = tuple(np.max(extents) / extents)
    print('empirical anisotropy of labeled objects = %s' % str(anisotropy))
    # 96 is a good default choice (see 1_data.ipynb)
    n_rays = 96

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    # use_gpu = False and gputools_available()
    use_gpu = False

    # Predict on subsampled grid for increased efficiency and larger field of view
    grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)

    # Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
    rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

    conf = Config3D (
        rays             = rays,
        grid             = grid,
        anisotropy       = anisotropy,
        use_gpu          = use_gpu,
        n_channel_in     = n_channel,
        # adjust for your data below (make patch size as large as possible)
        train_patch_size = (8,64,64),
        train_batch_size = 2,
    )
    print(conf)
    vars(conf)
    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        # limit_gpu_memory(0.8)
        # alternatively, try this:
        limit_gpu_memory(None, allow_growth=True)
    model = StarDist3D(conf, name='stardist', basedir='models')  
    median_size = calculate_extents(Y, np.median)
    fov = np.array(model._axes_tile_overlap('ZYX'))
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")
    if any(median_size > fov):
        print("WARNING: median object size larger than field of view of the neural network.")
    # Data Augmentation
    def random_fliprot(img, mask, axis=None): 
        if axis is None:
            axis = tuple(range(mask.ndim))
        axis = tuple(axis)
                
        assert img.ndim>=mask.ndim
        perm = tuple(np.random.permutation(axis))
        transpose_axis = np.arange(mask.ndim)
        for a, p in zip(axis, perm):
            transpose_axis[a] = p
        transpose_axis = tuple(transpose_axis)
        img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim))) 
        mask = mask.transpose(transpose_axis) 
        for ax in axis: 
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=ax)
                mask = np.flip(mask, axis=ax)
        return img, mask 

    def random_intensity_change(img):
        img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
        return img

    def augmenter(x, y):
        """Augmentation of a single input/label image pair.
        x is an input image
        y is the corresponding ground-truth label image
        """
        # Note that we only use fliprots along axis=(1,2), i.e. the yx axis 
        # as 3D microscopy acquisitions are usually not axially symmetric
        x, y = random_fliprot(x, y, axis=(1,2))
        x = random_intensity_change(x)
        return x, y

    # plot some augmented examples
    img, lbl = X[0],Y[0]
    plot_img_label(img, lbl)
    for _ in range(3):
        img_aug, lbl_aug = augmenter(img,lbl)
        plot_img_label(img_aug, lbl_aug, img_title="image augmented (XY slice)", lbl_title="label augmented (XY slice)")
    # Training

    quick_demo = False

    if quick_demo:
        print (
            "NOTE: This is only for a quick demonstration!\n"
            "      Please set the variable 'quick_demo = False' for proper (long) training.",
            file=sys.stderr, flush=True
        )
        model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter,
                    epochs=2, steps_per_epoch=5)

        print("====> Stopping training and loading previously trained demo model from disk.", file=sys.stderr, flush=True)
        model = StarDist3D.from_pretrained('3D_demo')
    else:
        model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter)

    # Threshold optimization
    if quick_demo:
        # only use a single validation image for demo
        model.optimize_thresholds(X_val[:1], Y_val[:1])
    else:
        model.optimize_thresholds(X_val, Y_val)
# Evaluation and Detection Performance
# Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
#               for x in tqdm(X_val)]
# plot_img_label(X_val[0],Y_val[0], lbl_title="label GT (XY slice)")
# plot_img_label(X_val[0],Y_val_pred[0], lbl_title="label Pred (XY slice)")
# taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for t in tqdm(taus)]
# stats[taus.index(0.7)]
# fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

# for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
#     ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
# ax1.set_xlabel(r'IoU threshold $\tau$')
# ax1.set_ylabel('Metric value')
# ax1.grid()
# ax1.legend()

# for m in ('fp', 'tp', 'fn'):
#     ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
# ax2.set_xlabel(r'IoU threshold $\tau$')
# ax2.set_ylabel('Number #')
# ax2.grid()
# ax2.legend();

