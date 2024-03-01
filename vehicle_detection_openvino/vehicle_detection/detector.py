#!/usr/bin/env python3
"""
 Copyright (C) 2018-2022 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import cv2

from openvino.model_zoo.model_api.models import OutputTransform
from vehicle_detection.draw_detection import draw_detections


def detect(capture, model, palette, pipeline):
    output_resolution = None
    output_transform = None

    next_frame_id = 0
    next_frame_id_to_show = 0

    while True:
        if pipeline.callback_exceptions:
            raise pipeline.callback_exceptions[0]
        # Process all completed requests
        results = pipeline.get_result(next_frame_id_to_show)
        if results:
            objects, frame_meta = results
            frame = frame_meta['frame']

            frame = draw_detections(frame, objects, palette, model.labels, output_transform)

            next_frame_id_to_show += 1

            # cv2.imshow('Detection Results', frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # continue

        if pipeline.is_ready():
            frame = capture.read()
            if frame is None:
                if next_frame_id == 0:
                    raise ValueError("Can't read an image from the input")
                break
            if next_frame_id == 0:
                output_transform = OutputTransform(frame.shape[:2], output_resolution)
                if output_resolution:
                    output_resolution = output_transform.new_resolution
                else:
                    output_resolution = (frame.shape[1], frame.shape[0])

            # Submit for inference
            pipeline.submit_data(frame, next_frame_id, {'frame': frame})
            next_frame_id += 1




