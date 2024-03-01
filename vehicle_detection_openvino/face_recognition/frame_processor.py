from face_recognition.helper.utils import crop
from face_recognition.helper.landmarks_detector import LandmarksDetector
from face_recognition.helper.face_detector import FaceDetector
from face_recognition.helper.faces_database import FacesDatabase
from face_recognition.helper.face_identifier import FaceIdentifier

import logging as log
import sys
from pathlib import Path
from openvino.runtime import Core

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))


no_show = None

match_algo = 'HUNGARIAN'

fg = ''
run_detector = None
allow_grow = None

m_fd = 'intel/models/face-detection-retail-0004/FP16/face-detection-retail-0004.xml'
m_lm = 'intel/models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml'
m_reid = 'intel/models/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml'
fd_input_size = (0, 0)

d_fd = 'CPU'
d_lm = 'CPU'
d_reid = 'CPU'

t_fd = 0.6
t_id = 0.3
exp_r_fd = 1.15


class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, ):
        self.allow_grow = allow_grow and not no_show

        log.info('OpenVINO Runtime')
        core = Core()

        self.face_detector = FaceDetector(core, m_fd,
                                          fd_input_size,
                                          confidence_threshold=t_fd,
                                          roi_scale_factor=exp_r_fd)
        self.landmarks_detector = LandmarksDetector(core, m_lm)
        self.face_identifier = FaceIdentifier(core, m_reid,
                                              match_threshold=t_id,
                                              match_algo=match_algo)

        self.face_detector.deploy(d_fd)
        self.landmarks_detector.deploy(d_lm, self.QUEUE_SIZE)
        self.face_identifier.deploy(d_reid, self.QUEUE_SIZE)

        log.debug('Building faces database using images from {}'.format(fg))
        self.faces_database = FacesDatabase(fg, self.face_identifier,
                                            self.landmarks_detector,
                                            self.face_detector if run_detector else None, no_show)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info('Database is built, registered {} identities'.format(len(self.faces_database)))

    def process(self, frame):
        orig_image = frame.copy()

        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE < len(rois):
            log.warning('Too many faces for processing. Will be processed only {} of {}'
                        .format(self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]

        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
        if self.allow_grow and len(unknowns) > 0:
            for i in unknowns:
                # This check is preventing asking to save half-images in the boundary of images
                if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
                        (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
                        (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
                    continue
                crop_image = crop(orig_image, rois[i])
                name = self.faces_database.ask_to_save(crop_image)
                if name:
                    face_identities[i].id = self.faces_database.dump_faces(crop_image, face_identities[i].descriptor, name)

        return [rois, landmarks, face_identities]
