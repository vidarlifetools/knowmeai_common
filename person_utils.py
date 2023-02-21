
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import torch
import cv2
import time
import json

COCO_PERSON_CLASS = 1

class PersonBbox:
    def __init__(self, tracking, tracking_bbox, tracking_first_frame, scale = 1.0, device = None, *args):
        self.people_detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
        )
        if device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = device
        self.people_detector.to(self.device)
        self.people_detector.eval()

        self.scale = scale
        self.tracking = tracking
        self.tracker = cv2.TrackerKCF_create()
        self.tracking_bbox = tracking_bbox
        self.tracking_first_frame = tracking_first_frame
        self.frame_no = 0

    def scale_image(self, image):
        # resize image
        width = int(image.shape[1] * self.scale)
        height = int(image.shape[0] * self.scale)
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized

    def scale_bbox(self, bbox):
        for i in range(len(bbox)):
            bbox[i] = int(bbox[i] * self.scale)
        return bbox

    def rescale_bbox(self, bbox):
        if bbox:
            for i in range(len(bbox)):
                bbox[i] = int(bbox[i] / self.scale)
        return bbox

    def detect(self, color, threshold = 0.5):
        color = self.scale_image(color)
        if self.tracking:
            if self.tracking_first_frame == self.frame_no:
                self.init_tracker(color, self.tracking_bbox)
            if self.frame_no >= self.tracking_first_frame:
                bbox = self.detect_target_person(color)
            else:
                bbox = self.detect_person(color, threshold=threshold)
        else:
            bbox = self.detect_person(color, threshold=threshold)
        bbox = self.rescale_bbox(bbox)
        self.frame_no += 1
        return bbox
    def detect_person(self, color, threshold=0.5):
        pil_image = Image.fromarray(color)  # Load the image
        transform = transforms.Compose(
            [transforms.ToTensor()]
        )  # Defing PyTorch Transform
        transformed_img = transform(pil_image)  # Apply the transform to the image
        pred = self.people_detector(
            [transformed_img.to(self.device)]
        )  # Pass the image to the model
        pred_classes = pred[0]["labels"].cpu().numpy()
        pred_boxes = [
            [i[0], i[1], i[2], i[3]]
            for i in list(pred[0]["boxes"].cpu().detach().numpy().astype(int))
        ]  # Bounding boxes
        pred_scores = list(pred[0]["scores"].cpu().detach().numpy())

        person_boxes = []
        # Select box has score larger than threshold and is person
        for pred_class, pred_box, pred_score in zip(
                pred_classes, pred_boxes, pred_scores
        ):
            if (pred_score > threshold) and (pred_class == COCO_PERSON_CLASS):
                person_boxes.append(pred_box)
        if len(person_boxes) > 0:
            return person_boxes[0]
        else:
            return None
    def init_tracker(self, first_frame, bbox):
        # Old SORT tracker
        #KalmanBoxTracker.count = 0
        #self.tracker = Sort(max_age=10, min_hits=2)
        #self.tracker.update(bbox)

        resized_img = self.scale_image(first_frame)
        resized_bbox = self.scale_bbox(bbox)
        # New KCF tracker initialise with the first frame and the bbox
        ok = self.tracker.init(resized_img, [resized_bbox[0], resized_bbox[1],
                                            resized_bbox[2]-resized_bbox[0],
                                            resized_bbox[3]-resized_bbox[1]])
        # If init_tracking is called, tracking is started
        self.tracking = True

    def detect_target_person(self, color):
        """Return bounding box of target person"""
        # Not needed any more because KCF tracker performs cross-correlation on the raw image
        # bboxes = self.detect_people(color, 0.9)
        # if not bboxes:
        #    return None

        if self.tracker is not None:
            # Old SORT tracker
            # b = self.tracker.update(np.array(bboxes))
            # bboxes = b[b[:, 4] == 1][:, 0:4].astype(int)

            # New KCF tracker
            bboxes = [self.tracker.update(color)[1]]
        else:
            return None

        #        if not len(bboxes):
        #            return None
        # in case if x1,y1,x2,y2 are 0s
        if sum(bboxes[0]) == 0:
            return None

        # limit bbox to image borders
        ymax, xmax = color.shape[:2]
        x1, y1, x2, y2 = bboxes[0]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x1 + x2, xmax)
        y2 = min(y1 + y2, ymax)

        return [x1, y1, x2, y2]
