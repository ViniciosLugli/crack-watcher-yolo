from roboflow import Roboflow
from ultralytics import YOLO
from dotenv import load_dotenv
import torch
import os
import gc


gc.collect()
torch.cuda.empty_cache()

load_dotenv()

PROJECT_NAME = "university-bswxt"
DATASET_NAME = "crack-bphdr"
MODEL_PATH = "crack-2/yolov8n.pt"
DATA_PATH = "crack-2/data.yaml"
ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY')


class RoboflowTrainer:
    def __init__(self, api_key, project_name, dataset_name, model_path):
        self.api_key = api_key
        self.project_name = project_name
        self.dataset_name = dataset_name
        self.model_path = model_path
        self.rf = Roboflow(api_key=self.api_key)
        self.model = None

    def download_dataset(self, version):
        project = self.rf.workspace(
            self.project_name).project(self.dataset_name)
        dataset = project.version(version).download("yolov8")
        return dataset

    def train_model(self, epochs, batch_size):
        self.model = YOLO(model=self.model_path)
        self.model.train(data=DATA_PATH, epochs=epochs, batch=batch_size)


def train():
    trainer = RoboflowTrainer(api_key=ROBOFLOW_API_KEY, project_name=PROJECT_NAME,
                              dataset_name=DATASET_NAME, model_path=MODEL_PATH)

    dataset = trainer.download_dataset(version=2)
    trainer.train_model(5, 8)


if __name__ == "__main__":
    train()
