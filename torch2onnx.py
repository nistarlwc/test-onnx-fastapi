# coding:utf-8
import os
import numpy as np
import torch, onnx
from onnxmltools.utils import float16_converter
from builders.DDRNetJIT import DDRNetJIT
from algo.config import config_data, load_param

model_info = load_param(config_data['GENERAL']['MODEL_CONFIG_PATH'])

"""######################################################"""
def segment2onnx():
    device = torch.device("cuda:%s" % config_data['GENERAL']['CUDA_VISIBLE_DEVICES'] if torch.cuda.is_available() else "cpu")
    # torch.cuda.current_device()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    BATCH_SIZE = config_data['SEGMENT']['BATCH_SIZE']
    SCORE_MAP_THRESH = config_data['SEGMENT']['SCORE_MAP_THRESH']
    SEG_SIZE_W = config_data['SEGMENT']['SEG_SIZE_W']
    SEG_SIZE_H = config_data['SEGMENT']['SEG_SIZE_H']
    CHECKPOINT_DIR = model_info['SEGMENT_CHECKPOINT_DIR']
    output_path = os.path.dirname(CHECKPOINT_DIR)

    model = DDRNetJIT(pretrained=False, num_classes=config_data['SEGMENT']['NUM_CLASSES'])
    checkpoint = torch.load(CHECKPOINT_DIR)['model']  # , map_location=device
    model.load_state_dict(checkpoint)
    model = model.eval()
    model = model.to(device)

    image = np.random.randint(low=0, high=255, size=(SEG_SIZE_H, SEG_SIZE_W, 3), dtype=np.uint8)
    im_cpu = torch.from_numpy(image)
    im_cpu = im_cpu.unsqueeze(0)
    im_gpu = im_cpu.to(device)
    with torch.no_grad():
        model_trace = torch.jit.trace(model, im_gpu)

    onnx_path = os.path.join(output_path, 'segment.onnx')
    torch.onnx.export(model_trace, im_gpu, onnx_path, input_names=['input'], output_names=['output'],
                      export_params=True, training=torch.onnx.TrainingMode.EVAL, do_constant_folding=True, opset_version=15)

    onnx_model = onnx.load_model(onnx_path)
    # onnx.checker.check_model(onnx_model)
    trans_model = float16_converter.convert_float_to_float16(onnx_model, keep_io_types=True)
    onnx_fp16_path = os.path.join(output_path, 'segment_fp16.onnx')
    onnx.save_model(trans_model, onnx_fp16_path)


    os.remove(onnx_path)

if __name__ == '__main__':
    segment2onnx()