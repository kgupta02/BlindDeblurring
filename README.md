# BlindDeblurring
Image Debarring using classical image processing techniques (IN PROGRESS)

<pre> ```
BlindDeblurring/
│
├── README.md
├── requirements.txt         # or environment.yml for conda
├── setup.py                 # optional if you want to make it installable
│
├── data/                    # optional: store example input images or test images
│   ├── sample1.png
│   ├── ...
│
├── results/                 # store output images and kernels
│   ├── output_deblurred.png
│   ├── kernel_estimated.png
│
├── src/                  
│   ├── __init__.py
│   │
│   ├── main_deblurring.py   # main script/entry point
│   │
│   ├── io_utils.py          # handles input/output (reading, saving images, etc.)
│   ├── pyramid.py           # builds coarse-to-fine image pyramids
│   ├── blur_model.py        # defines the blur model and latent map M
│   ├── priors.py            # hyper-Laplacian prior & gradient utilities
│   ├── optimization.py      # core iterative optimization logic
│   ├── nonblind_deconv.py   # optional: final non-blind deconvolution
│   └── metrics.py           # functions for PSNR, SSIM, etc. (if needed)
│
└── tests/                   # optional: unit tests or integration tests
    ├── test_optimization.py
    └── ...
``` </pre>
