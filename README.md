# BlindDeblurring
ECE 556 Final Project: Image Deblurring using classical image processing techniques

Implements "Blind Deblurring for Saturated Images" by Liang Chen, et. al in Python

Image Data retrieved from: https://drive.google.com/file/d/1Yyyub_ylDY5IXfE57DvsdecG7HlkSFBS/view

<pre> ```
BlindDeblurring/
│
├── README.md
│
├── data/                    # test images
│   ├── [FILE_NAME].png
│   ├── ...
│
├── results/                 # store output images and kernels
│   ├── output_deblurred.png
│   ├── kernel_estimated.png
│
└── src/                  
    ├── __init__.py
    │
    ├── main_deblurring.py   # main script/entry point
    │
    ├── io_utils.py          # handles input/output
    ├── pyramid.py           # builds coarse-to-fine image pyramids
    ├── blur_model.py        # defines the blur model and latent map M
    ├── priors.py            # hyper-Laplacian prior & gradient utilities
    ├── optimization.py      # core iterative optimization logic
    ├── nonblind_deconv.py   # final non-blind deconvolution
    └── metrics.py           # functions for PSNR, SSIM, etc.
``` </pre>
