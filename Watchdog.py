import cv2
import onnxruntime as ort
import numpy as np
import os
import pickle
from clip_tokenizer import tokenize



import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


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

    def generate_imagedir_features_file(self, imgname_new,image_dir):
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
        top5_imglist = [(imglist[similar_id[i]], softmax_logit[similar_id[i]]) for i in range(len(similar_id))]
        return top5_imglist

with open('features3.pkl', 'rb') as f:
    q = pickle.load(f)
print(q)

mynet = Clip('./image_model.onnx', './text_model.onnx')
def detect(new_file_name):
    for i in range(3):
        try:
            image_dir = os.path.join('C:\\', 'testimgs')
            image_features, imglist = mynet.generate_imagedir_features_file(new_file_name, image_dir)
            with open('features3.pkl', 'wb') as f:
                pickle.dump((image_features, imglist), f)
            print('生成特征向量数据库成功!!!')
        except:
            print(f"第{i+1}次尝试失败...")
            time.sleep(60)

# 定义事件处理器类

class MyHandler(FileSystemEventHandler):

    def on_created(self, event):
        if event.is_directory:
            print(event.src_path)
        else:
            path = event.src_path
            #fileName = os.path.basename(path)
            detect(str(path))
            print(f'File created: {path}')

    def on_moved(self, event):
        if event.is_directory:
            what = "Directory"
        else:
            what = "File"
        print(f'{what} moved: {event.src_path} to {event.dest_path}')

    def on_deleted(self, event):
        if event.is_directory:
            print(f'目录被删除: {event.src_path}')
        else:
            try:
                filename = event.src_path

                # 读取现有特征数据库
                with open('features3.pkl', 'rb') as f:
                    image_features, imglist = pickle.load(f)
                
                print(f"当前数据库中的文件列表: {imglist}")
                print(f"要删除的文件: {filename}")
                
                # 找到被删除文件的索引
                if filename in imglist:
                    idx = imglist.index(filename)
                    # 删除对应的特征和文件名
                    image_features = np.array(image_features)  # 确保是numpy数组
                    image_features = np.delete(image_features, idx, axis=0)
                    imglist.pop(idx)
                    
                    # 保存更新后的数据库
                    with open('features3.pkl', 'wb') as f:
                        pickle.dump((image_features, imglist), f)
                    print(f'文件 {filename} 的特征已从数据库中删除')
                else:
                    print(f'文件 {filename} 在数据库中未找到')
                
            except Exception as e:
                print(f'更新特征数据库失败: {str(e)}')
            
            print(f'文件被删除: {event.src_path}')

# 初始化事件处理器实例
event_handler = MyHandler()

# 指定要监控的文件夹路径
watched_dir = 'C:\\testimgs'

# 初始化观察器
observer = Observer()

# 将事件处理器关联到观察器
observer.schedule(event_handler, watched_dir, recursive=True)

# 启动观察器
observer.start()

print(f'Starting to watch {watched_dir} for file system events...')
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()









