#!/usr/bin/env python3

from openvino.model_zoo.model_api.models import DetectionModel
from openvino.model_zoo.model_api.pipelines import get_user_config, AsyncPipeline
from openvino.model_zoo.model_api.adapters import create_core, OpenvinoAdapter, OVMSAdapter

import logging as log
import sys
from pathlib import Path

from visualizers import ColorPalette

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))


log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

# General Arguments
model = "intel/models/vehicle-detection-0202/FP16/vehicle-detection-0202.xml"
architecture_type = "ssd"
adapter = "openvino"  # choices=('openvino', 'ovms')
device = "CPU"

# Common model options Arguments
labels = None
prob_threshold = 0.5
resize_type = None  # choices=RESIZE_TYPES.keys()
input_size = (600, 600)
anchors = None
masks = None
layout = None
num_classes = None  # int

# Inference options Arguments
num_infer_requests = 0  # int
num_streams = ''
num_threads = None  # int

# Input/output options Arguments
loop = False
output = 'result.mp4'
output_limit = 1000
no_show = None
output_resolution = None  # (1280, 720)
utilization_monitors = ''

# Input transform options Arguments
reverse_input_channels = False
mean_values = None  # Example: 255.0 255.0 255.0
scale_values = None  # Example: 255.0 255.0 255.0

# Debug options Arguments
raw_output_message = False

# Execution process start
model_adapter = None

if adapter == 'openvino':
    plugin_config = get_user_config(device, num_streams, num_threads)
    model_adapter = OpenvinoAdapter(create_core(), model, device=device, plugin_config=plugin_config,
                                    max_num_requests=num_infer_requests, model_parameters={'input_layouts': layout})
elif adapter == 'ovms':
    model_adapter = OVMSAdapter(model)

configuration = {
    'resize_type': resize_type,
    'mean_values': mean_values,
    'scale_values': scale_values,
    'reverse_input_channels': reverse_input_channels,
    'path_to_labels': labels,
    'confidence_threshold': prob_threshold,
    'input_size': input_size,  # The CTPN specific
    'num_classes': num_classes,  # The NanoDet and NanoDetPlus specific
}


class Configuration:
    model = DetectionModel.create_model(architecture_type, model_adapter, configuration)
    detector_pipeline = AsyncPipeline(model)
    palette = ColorPalette(len(model.labels) if model.labels else 100)
