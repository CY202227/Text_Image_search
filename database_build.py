import cv2
import onnxruntime as ort
import numpy as np
import os
import pickle
from clip_tokenizer import tokenize


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
        
        # 递归遍历文件夹及子文件夹
        def walk_dir(dir_path):
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    # 检查文件扩展名是否为图片
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        path = os.path.join(root, file)
                        print(f"处理图片: {path}")
                        srcimg = cv2.imread(path)
                        if srcimg is None:
                            print(f"无法读取图片: {path}")
                            continue
                        img_feat = self.generate_image_feature(srcimg)
                        image_features.append(img_feat.squeeze())
                        imglist.append(path)
        
        walk_dir(image_dir)
        
        if not image_features:
            raise ValueError("未找到任何有效的图片文件")
        
        image_features = np.stack(image_features, axis=0)
        return image_features, imglist

    def generate_imagedir_features_file(self, imgname_new,image_dir):
        print("检测到新增文件，正在重新建立数据库")
        print(imgname_new)
        with open('features.pkl', 'rb') as f:
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
        top5_imglist = [(imglist[similar_id[i]], softmax_logit[similar_id[i]]) for i in range(len(similar_id))]
        return top5_imglist

mynet = Clip('./image_model.onnx', './text_model.onnx')


print('正在生成特征向量数据库!!!')
# 输入文件夹，生成图片的特征向量，保存到数据库文件

image_dir = os.path.join('C:\\', 'testimgs')
print(str(image_dir))
image_features, imglist = mynet.generate_imagedir_features(image_dir)  # 只需要运行一次,除非出现异常，否则注释掉即可

with open('features3.pkl', 'wb') as f:
    pickle.dump((image_features, imglist), f)
    print(imglist)
print('生成特征向量数据库成功!!!')