# -*- coding: UTF-8 -*-
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import os, time, base64, sys, socket, _thread, shutil, datetime
import traceback
from algo.config import config_data, load_param
from algo.inference_onnx import *
import uvicorn
from fastapi import FastAPI, Request
app = FastAPI()

@app.post('/algorithm/api/GDS1/detect')
def detect_GDS1(data: dict):
    code = 0  # 错误码
    message = "Success"
    response_data = {}

    try:
        start_time = time.time()
        image_path = data["image_path"]

        # 3）读取图像,根据图像传输模式，进行数据加载
        cv_image = None  # 初始化
        try:
            print('Read image')
            cv_image = cv2.imread(image_path, -1)
        except:
            print("Get Image error!!!")
            return {"code": code, "message": "Get Image error!!!", "data": {}}

        if cv_image.shape[0] != config_data['SEGMENT']['SEG_SIZE_H'] or cv_image.shape[1] != config_data['SEGMENT']['SEG_SIZE_W']:
            cv_image = cv2.resize(cv_image, (config_data['SEGMENT']['SEG_SIZE_W'], config_data['SEGMENT']['SEG_SIZE_H']))

        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        # print("%s: HTTP GET_DATA, RUN TIME IS:  %s" % (os.path.basename(image_path), total_time))

        start_time = time.time()
        predict = prediction(cv_image)
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        print(os.path.basename(image_path) + ": Segment RUN TIME IS:  %s" % total_time)

    except Exception as e:
        code = 101
        message = "program error: %s" % str(e)
        response_data = {}

    return {
        "code": code,
        "message": message,
        "data": response_data
    }


if __name__ == '__main__':
    host = config_data["GENERAL"]['HOST']
    port = int(config_data["GENERAL"]['PORT'][0])
    print('Init Parser...')
    uvicorn.run(app, host=host, port=port)