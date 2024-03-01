
import cv2
from openvino.model_zoo.model_api.models import DetectionWithLandmarks


def draw_detections(frame, detections, palette, labels, output_transform):
    frame = output_transform.resize(frame)

    print("--------------------------")
    print(len(detections))
    print("--------------------------")

    for detection in detections:
        class_id = int(detection.id)
        color = palette[class_id]
        det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
        xmin, ymin, xmax, ymax = detection.get_coords()
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
                    (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

        if isinstance(detection, DetectionWithLandmarks):
            for landmark in detection.landmarks:
                landmark = output_transform.scale(landmark)
                cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 255), 2)
    return frame
