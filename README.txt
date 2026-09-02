================================================================================
                    OCMTCN - Automatic Logging Lithology Identification System
================================================================================
Project Overview
--------
This project is built on the PyTorch deep‑learning framework and implements OCMTCN (Order‑Consistent Multi‑Scale Temporal Convolutional Network), a model for automatic stratigraphic lithology classification using well‑logging data.

Project Structure
--------
OCMTCN/
├── main.py                 # Main program entry; contains training, validation and testing workflows
├── __init__.py
├── models/
│   └── OCMTCN.py          # OCMTCN model definition (including monotonic self‑attention, CRF, etc.)
├── OtherFile/
│   ├── config.yaml        # Configuration file (hyperparameters, data paths, etc.)
│   ├── Tools.py           # Configuration parsing utilities
│   ├── Tool2.py           # Data‑processing utilities
│   ├── ModelTools.py      # Model‑evaluation utilities (accuracy, loss functions, etc.)
│   ├── PlaintTools.py     # Visualization and plotting utilities
│   └── default.py         # Default configuration

Configuration Instructions
--------
The configuration file is located at OtherFile/config.yaml. Key parameters are listed below:
    input_path:       Directory path for well‑logging datasets
    input_feature:    Names of input feature columns (e.g., GR, AC, QT, CNL)
    window_size:      Sliding‑window size (default: 1024)
    stride:           Sliding‑window stride (default: 256)
    batch_size:       Batch size (default: 32)
    epochs:           Maximum training epochs (default: 500)
    patience:         Early‑stopping patience value (default: 30)
    n_splits:         Number of folds for K‑fold cross‑validation (default: 5)
    model:            Model type ("OCMTCN")
    seed:             Random seed (default: 0)
    is_plaint:        Toggle visualization output (true/false)
    is_add_loss:      Toggle reverse‑order loss term (true/false)
    is_plus:          Toggle data augmentation (true/false)
    is_muti:          Toggle multi‑resolution feature fusion (true/false)
    pic_out_path:     Output directory for visualization figures

Execution
--------
Run the main program directly:
    python main.py
Or specify the target GPU device:
    CUDA_VISIBLE_DEVICES=0 python main.py

Output Description
--------
Upon training completion, console outputs include:
    - Per‑epoch loss and GPU runtime
    - Validation‑set accuracy
    - Test‑set comprehensive metrics (accuracy, precision, recall, F1‑score, Kappa coefficient)
    - Per‑lithology F1‑score and sample counts

If is_plaint is enabled, figures will be generated under pic_out_path:
    └── visualization_results/
        ├── WellName1/
        │   ├── WellName1_Segment_1.png
        │   └── ...
        └── WellName2/
            └── ...
