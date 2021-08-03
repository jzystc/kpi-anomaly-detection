import os
os.system('pip uninstall -y enum34 && pip install optuna')
from lgbm_train import train
from lgbm_test import detect

if __name__ == "__main__":
    
    train("data", "train")
    detect("data", "test")