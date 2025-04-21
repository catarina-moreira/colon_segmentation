# colon_segmentation


colon_segmentation_project/
│
├── src/                           # Source code
│   ├── train.py                   # Main training script (colon-segmentation.py)
│   ├── ensemble.py                # Ensemble model implementation (ensemble-model.py)
│   ├── inference.py               # Inference and evaluation script (inference-script.py)
│   ├── utils/                     # Utility functions
│   │   ├── __init__.py
│   │   ├── data_utils.py          # Data loading and preprocessing utilities
│   │   ├── model_utils.py         # Model-related utilities
│   │   └── visualization.py       # Visualization functions
│   │
│   └── models/                    # Model definitions
│       ├── __init__.py
│       └── network.py             # Network architectures
│
├── configs/                       # Configuration files
│   ├── train_config.yaml          # Training configuration
│   ├── ensemble_config.yaml       # Ensemble configuration
│   └── inference_config.yaml      # Inference configuration
│
├── data/                          # Data directory
│   └── Task10_Colon/              # Colon dataset
│       ├── imagesTr/              # Training images
│       ├── labelsTr/              # Training labels
│       ├── imagesTs/              # Test images
│       └── labelsTs/              # Test labels (if available)
│
├── outputs/                       # Output directory
│   ├── models/                    # Saved models
│   │   ├── dynunet/               # DynUNet model checkpoints
│   │   ├── unet/                  # UNet model checkpoints
│   │   └── basicunet/             # BasicUNet model checkpoints
│   │
│   ├── predictions/               # Prediction outputs
│   │   ├── single_model/          # Single model predictions
│   │   └── ensemble/              # Ensemble predictions
│   │
│   └── visualizations/            # Visualization outputs
│       ├── training/              # Training visualizations
│       └── inference/             # Inference visualizations
│
├── notebooks/                     # Jupyter notebooks for exploration and analysis
│   ├── data_exploration.ipynb     # Dataset exploration
│   └── results_analysis.ipynb     # Analysis of results
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── run.sh                         # Shell script to run the pipeline

