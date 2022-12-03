# Usage:
    1. Create conda environmen
        conda create -n deepfl_eval python=3.9.12
    2. Activate conda environmen
        conda activate deepfl_eval
    3. Cd to requirements
        cd requirements
    4. Install all required packages
        bash requirements.sh
    5. Download weights and put to corresponding folders
        Links of weights are in models/link.txt
    6. Download PCA models and put to corresponding folders
        Links of weights are in models/link.txt
    7. Download GA indices and put to corresponding folders
        Links of indices are in models/link.txt
    8. Download GA weights and put to corresponding folders
        Links of indices are in models/link.txt    
    
# To Use Evaluater for a single image run DeepFL.py 
        Giving the path of image
            -d (--data_path) path_of_image
        For visualizing
            -v (--visualize) True
        For prediction with PCA
            -p (--pca) True
        For prediction with Genetic algorithm
            -ga (--genetic_algorithm) True
        *NOTE: Default predicts with GA indices
        *NOTE: Default value for pca is False, default value for ga is True
        *NOTE: Don't pass -p True and -ga True (or -p False and -ga False) this will raise errors
        
# To Use Evaluater for all benchmarks
        Open jupyter notebook and run Evaluator.ipynb
