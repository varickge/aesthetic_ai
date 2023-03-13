# Usage:
    1. Create conda environment
        conda create -n deepfl_eval python=3.9.12
    2. Activate conda environment
        conda activate deepfl_eval
    3. Change directory to requirements
        cd requirements
    4. Install all required packages
        bash requirements.sh
    5. Download weights and put to corresponding folders
        Links of weights are in models/link.txt
    
# To Use Evaluater for a single image run DeepFL.py 
        Giving the path of image
            -d (--data_path) path_of_image
        For visualizing
            -v (--visualize) True
     
        *NOTE: Default predicts with GA indices trained on all_resied 996 images
        
# To Use Evaluater for all benchmarks
        *NOTE: Benchmark data must be in Data/ folder, Data/ and DeepFL_CLI_evaluating/ folders must be in the same directory. 
               Check before running
        *NOTE: Download data from https://www.dropbox.com/s/jfqk1jlcsbwuuvu/AesthAI.zip?dl=0 path and unzip in Data/ folder      
        Open jupyter notebook and run Evaluator.ipynb
        