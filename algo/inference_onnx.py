# coding:utf-8

import os,time,cv2,json
import numpy as np

import onnxruntime as rt

from algo.config import config_data, load_param

"""参数配置"""
model_info = load_param(config_data['GENERAL']['MODEL_CONFIG_PATH'])
os.environ['CUDA_VISIBLE_DEVICES'] = config_data['GENERAL']['CUDA_VISIBLE_DEVICES']

BATCH_SIZE = config_data['SEGMENT']['BATCH_SIZE']
SEG_SIZE_W = config_data['SEGMENT']['SEG_SIZE_W']
SEG_SIZE_H = config_data['SEGMENT']['SEG_SIZE_H']

class SemanticSegment(object):
    def __init__(self):
        CHECKPOINT_DIR = model_info['SEGMENT_CHECKPOINT_DIR']
        if CHECKPOINT_DIR[-5:] != '.onnx':
            model_path = os.path.dirname(model_info['SEGMENT_CHECKPOINT_DIR'])
            CHECKPOINT_DIR = os.path.join(model_path, "segment_fp16.onnx")
        start_time = time.time()
        sess_providers = ['CUDAExecutionProvider']
        sess_options = rt.SessionOptions()
        self.session = rt.InferenceSession(CHECKPOINT_DIR, sess_options, sess_providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.num_classes = config_data['SEGMENT']['NUM_CLASSES']

        zero_image = np.random.randint(low=0, high=255, size=(BATCH_SIZE, SEG_SIZE_H, SEG_SIZE_W, 3), dtype=np.uint8)
        _ = self.session.run(None, {self.input_name: zero_image})
        zero_image = np.random.randint(low=0, high=255, size=(BATCH_SIZE, SEG_SIZE_H, SEG_SIZE_W, 3), dtype=np.uint8)
        _ = self.session.run(None, {self.input_name: zero_image})
        zero_image = np.random.randint(low=0, high=255, size=(BATCH_SIZE, SEG_SIZE_H, SEG_SIZE_W, 3), dtype=np.uint8)
        _ = self.session.run(None, {self.input_name: zero_image})
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        print("model.eval RUN TIME IS:  %s" % total_time)

    def predict_label(self, image):
        input_tensor = np.expand_dims(image, axis=0)
        prediction = self.session.run(None, {self.input_name: input_tensor})[0]
        pred = prediction[0][0].astype(np.uint8)
        return pred

semantic_segment_instance = SemanticSegment()

def prediction(image):
    return semantic_segment_instance.predict_label(image)