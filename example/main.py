from naie_train import train
from naie_detect import detect

if __name__ == "__main__":
    train("DatasetService", "KPI_TRAIN")
    detect("DatasetService", "KPI_TEST")