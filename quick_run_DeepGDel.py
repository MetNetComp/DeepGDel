import os
import sys
from DeepGDel_main import main  

def run():
    sys.argv = ['quick_run.py', '--CBM', 'e_coli_core', '--use_cpu', '1']  # Simulate command-line arguments
    main()  # Call the main function from DeepGDel_main.py

if __name__ == "__main__":
    run()
