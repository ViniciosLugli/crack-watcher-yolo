from ultralytics import YOLO
import cv2
import sys


class CrackModelRunner:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = YOLO(model=self.model_path)

    def find(self, image_path):
        image = cv2.imread(image_path)

        return self.model.predict(image)

    def draw(self, image_path, results):
        image = cv2.imread(image_path)

        for result in results:
            boxes = result.boxes.cpu().numpy()

            for box in boxes:
                r = box.xyxy[0].astype(int)

                cv2.rectangle(image, r[:2], r[2:], (255, 255, 255), 2)

                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(
                    image,
                    "Crack",
                    (r[0], r[1] - 10),
                    font,
                    1.0,
                    (255, 255, 255),
                    1,
                )

        return image


def run(image_path):
    image_path = str(image_path)
    runner = CrackModelRunner("models/yolov8n.pt")
    results = runner.find(image_path)
    image = runner.draw(image_path, results)
    cv2.imshow("Crack Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run("models/tests/0.jpg")
