import os
from baseline.baseline import run_baseline
from utils.evaluations import *

def main():
    CBM_list = ['e_coli_core']  # List of models to run
    M = 100  # Number of random baseline runs
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    
    for CBM in CBM_list:
        relationship_folder_path = os.path.join(script_dir, 'Data', CBM, 'Label')
        print(f"Running baseline for {CBM} ...")
        run_baseline(CBM, M, relationship_folder_path)
    
if __name__ == "__main__":
    main()
