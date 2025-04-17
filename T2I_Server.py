from datetime import timedelta

import cv2
import onnxruntime as ort
import numpy as np
import os
import shutil
import pickle
from flask import Flask, jsonify, request
from clip_tokenizer import tokenize
import cn_clip.clip as clip
import torch
from PIL import Image
from cn_clip.clip import load_from_name

app = Flask(__name__)
app.config['app.json.ensure_ascii'] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(hours=1)
old_imglist = []


class Clip():
    def __init__(self, image_modelpath, text_modelpath):
        self.img_model = cv2.dnn.readNet(image_modelpath)
        self.input_height, self.input_width = 224, 224

        self.mean = np.array([0.48145466, 0.4578275, 0.40821073],
                             dtype=np.float32).reshape((1, 1, 3))
        self.std = np.array([0.26862954, 0.26130258, 0.27577711],
                            dtype=np.float32).reshape((1, 1, 3))

        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.txt_model = ort.InferenceSession(text_modelpath, so)
        self.context_length = 52

    def preprocess(self, srcimg):
        img = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height),
                         interpolation=cv2.INTER_CUBIC)
        img = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        return img

    def generate_image_feature(self, srcimg):
        img = self.preprocess(srcimg)
        blob = cv2.dnn.blobFromImage(img)
        self.img_model.setInput(blob)
        image_features = self.img_model.forward(self.img_model.getUnconnectedOutLayersNames())[0]

        img_norm = np.linalg.norm(image_features, axis=-1, keepdims=True)
        image_features /= img_norm
        return image_features

    def generate_text_feature(self, input_text):
        text = tokenize(input_text, context_length=self.context_length)
        text_features = []
        for i in range(len(text)):
            one_text = np.expand_dims(text[i], axis=0)
            text_feature = self.txt_model.run(None, {self.txt_model.get_inputs()[0].name: one_text})[0].squeeze()
            text_features.append(text_feature)
        text_features = np.stack(text_features, axis=0)
        txt_norm = np.linalg.norm(text_features, axis=1, keepdims=True)
        text_features /= txt_norm
        return text_features

    def run_image_classify(self, image, input_strs):
        image_features = self.generate_image_feature(image)
        text_features = self.generate_text_feature(input_strs)
        logits_per_image = 100 * np.dot(image_features, text_features.T)
        exp_logits = np.exp(logits_per_image - np.max(logits_per_image, axis=-1, keepdims=True))
        softmax_logit = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        max_str = input_strs[softmax_logit.argmax()]
        max_str_logit = softmax_logit.max()
        return max_str, max_str_logit

    def generate_imagedir_features(self, image_dir):
        imglist, image_features = [], []
        for imgname in os.listdir(image_dir):
            srcimg = cv2.imread(os.path.join(image_dir, imgname))
            if srcimg is None:  # 有可能存在不是图片的
                continue
            img_feat = self.generate_image_feature(srcimg)
            image_features.append(img_feat.squeeze())
            imglist.append(imgname)

        image_features = np.stack(image_features, axis=0)
        return image_features, imglist

    def ogenerate_imagedir_features_file(self, imgname_new, image_dir):
        print("检测到新增文件，正在重新建立数据库")
        print(imgname_new)
        with open('features3.pkl', 'rb') as f:
            image_features, imglist = pickle.load(f)
        imglist = list(imglist)
        print(image_features)
        srcimg = cv2.imread(os.path.join(image_dir, imgname_new))
        if srcimg is None:  # 有可能存在不是图片的
            pass
        img_feat = self.generate_image_feature(srcimg)

        new_image_features = np.array(img_feat.squeeze())
        image_features = list(image_features)
        image_features.append(new_image_features)
        print(image_features)
        imglist.append(str(imgname_new))

        image_features = np.stack(image_features, axis=0)
        return image_features, imglist

    def input_text_search_image(self, input_text, image_features, imglist):
        text_features = self.generate_text_feature(input_text)
        logits_per_image = 100 * np.dot(text_features, image_features.T)
        exp_logits = np.exp(logits_per_image - np.max(logits_per_image, axis=-1, keepdims=True))
        softmax_logit = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        softmax_logit = softmax_logit.reshape(-1)  # 拉平数组
        similar_id = np.argsort(-softmax_logit)  # 降序排列
        # top5_imglist = [(imglist[similar_id[i]], softmax_logit[similar_id[i]]) for i in range(10)]
        top5_imglist = [(imglist[similar_id[i]]) for i in range(5)]
        return top5_imglist


mynet = Clip('./image_model.onnx', './text_model.onnx')


###############################################################################################################
# print('正在生成特征向量数据库!!!')
# 输入文件夹，生成图片的特征向量，保存到数据库文件

# image_dir = os.path.join(os.getcwd(), 'testimgs')
# print(str(image_dir))
# image_features, imglist = mynet.generate_imagedir_features(image_dir)  # 只需要运行一次,除非出现异常，否则注释掉即可

# with open('features.pkl', 'wb') as f:
#    pickle.dump((image_features, imglist), f)
#    print(imglist)
# print('生成特征向量数据库成功!!!')

################################################################################################################

@app.route('/text_match_image', methods=['POST', 'GET'])
def text_match_image():  # upload face to face database
    print("Main System Activating text_match_image Mode")

    print('开始计算最相似的图片!')
    input_text = request.form.get('text')
    with open('features3.pkl', 'rb') as f:
        image_features, imglist = pickle.load(f)
    print('图片列表一共有' + str(len(imglist)) + '张')
    print('图片特征一共有' + str(len(image_features)) + '个')
    top5_imglist = mynet.input_text_search_image(input_text, image_features, imglist)
    print('返回的图片一共有' + str(len(top5_imglist)) + '张')

    image_dir = os.path.join('C:\\', 'testimgs')
    result_imgs = os.path.join(os.getcwd(), 'result_imgs')
    if os.path.exists(result_imgs):
        shutil.rmtree(result_imgs)
    os.makedirs(result_imgs)
    # for imgname, conf in top5_imglist:
    for imgname in top5_imglist:
        shutil.copy(os.path.join(image_dir, imgname), result_imgs)
    return str(top5_imglist)


#####输入提示词, 做图片分类

@app.route('/image_classifier', methods=['POST', 'GET'])  # 用于给图片进行分类
def detect():
    try:
        files = request.files.getlist('files')  # 获取文件列表
        uploaded_file_folder = './caches'  # 将图片存在caches文件夹中
        label = request.form.get('text')  # 获取分类标签
        label_text = ' ' + label
        label_chars = label_text.split('，')
        label_chars = list(label_chars)  # 将分类标签转为列表
        print(label_chars)
        result = []
    except ValueError:
        return jsonify({
            'msg': "上传失败",
            'code': 500,
            'data': {
                'Status': 'failed',
            }})
    try:
        for file in files:
            filename = file.filename
            save_path = os.path.join(uploaded_file_folder, filename)
            file.save(save_path)  # 保存在caches文件夹中
            # Label_probs = image_classifier(save_path, label_chars)  # 对图片进行分类
            # print("Label probs:", Label_probs)
            # max_index = np.argmax(np.array(Label_probs))
            # print(max_index)
            # answer = label_chars[max_index]  # 筛选出符合要求的结果
            # result.append(filename)  # 将返回结果保存
            mynet = Clip("image_model.onnx", "text_model.onnx")

            srcimg = cv2.imread(save_path)
            max_str, max_str_logit = mynet.run_image_classify(srcimg, label_chars)
            print(f"最大概率：{max_str_logit}, 对应类别：{max_str}")

        return jsonify({
            'msg': "搜索成功",
            'code': 200,
            'data': {
                'Status': 'success',
                'Result': '最大概率：' + str(max_str_logit),
                'Text': '分类结果为：' + str(max_str)
            }})
    except RuntimeError:
        return jsonify({
            'msg': "搜索失败",
            'code': 500,
            'data': {
                'Status': 'failed',
            }})


if __name__ == '__main__':
    # server = pywsgi.WSGIServer(('10.1.1.202', 5000), app)
    # server.serve_forever()

    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True, processes=1)
