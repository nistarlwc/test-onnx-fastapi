# coding: utf-8
import os, json, requests, time, base64, cv2, datetime, threading
import numpy as np

config_path = './config.json'
config_data = json.loads(open(config_path).read()) #, encoding='utf-8'
BATCH_SIZE = config_data['SEGMENT']['BATCH_SIZE']
SEG_SIZE_W = config_data['SEGMENT']['SEG_SIZE_W']
SEG_SIZE_H = config_data['SEGMENT']['SEG_SIZE_H']

def show_files(path, all_files):
    file_list = os.listdir(path)
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            show_files(cur_path, all_files)
            pass
        else:
            if cur_path.endswith('.bmp') or cur_path.endswith('.jpg') or cur_path.endswith('.png'):
                all_files.append(cur_path)

    return all_files

def tickFunc(image_list, idx, request_url):
    start2 = time.time()
    image_file = image_list[idx]
    req_json = {"image_path": image_file}
    response_return = requests.post(request_url, json=req_json).json()
    if response_return["code"] == 101:
        print("--- !!! error: %d" % response_return["code"])

if __name__ == '__main__':
    test_host = config_data["GENERAL"]['HOST']
    test_port = config_data["GENERAL"]['PORT']
    request_url_1 = "http://{}:{}/algorithm/api/GDS1/detect".format(test_host, int(test_port[0]))
    # image_list = show_files(image_folder, [])
    image_folder = 'img'
    os.makedirs(image_folder, exist_ok=True)
    image_list = []
    for i in range(50):
        image = np.random.randint(low=0, high=255, size=(SEG_SIZE_H, SEG_SIZE_W, 3), dtype=np.uint8)
        image_path = os.path.join(image_folder, "%d.png"%i)
        cv2.imwrite(image_path, image)
        image_list.append(image_path)
    print("image_list: ", len(image_list))

    start_time = time.time()
    for idx, image in enumerate(image_list):
        tickFunc(image_list, idx, request_url_1)

    time.sleep(10)

    for idx, image in enumerate(image_list):
        tickFunc(image_list, idx, request_url_1)

    # threads = []
    # num_threads = len(image_list)
    # for i in range(num_threads):
    #     thread = threading.Thread(target=tickFunc, args=(image_list, i, request_url_1,))
    #     threads.append(thread)  #
    # start_time = time.time()
    # for i in range(num_threads):
    #     threads[i].start()
    #     time.sleep(1 / 100.0)
    # for i in range(num_threads):
    #     threads[i].join()

    end_time = time.time()
    total_time = (end_time - start_time) * 1000
    print("RUN TIME IS:  %s" % total_time, total_time / len(image_list))
