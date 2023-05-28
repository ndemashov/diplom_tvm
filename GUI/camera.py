import cv2
import tkinter as tk
from tkinter import *
from PIL import Image,ImageTk
from datetime import datetime
from tkinter import messagebox, filedialog

import tvm
from tvm import relax
import numpy as np
from torchvision import transforms
from sklearn import metrics
import pickle
import os

from tvm.script.parser import ir as I
from tvm.script import relax as R

from tvm.relax.training import SetupTrainer
from tvm.relax.training.trainer import Trainer
from tvm.relax.training.optimizer import Adam
from tvm.relax.training.loss import CategoricalCrossEntropyLoss

def createwidgets():
    root.cameraLabel = Label(root, bg="#21556e", borderwidth=3, relief="groove")
    root.cameraLabel.grid(row=2, column=1, padx=10, pady=10, columnspan = 2)

    root.changeModel = Button(root, text="Change Model", command=changeModel, bg="#7b9eb0", font=('Shentox',15), width=15)
    root.changeModel.grid(row=6, column=1, padx=10, pady=10, columnspan = 2)

    root.predictedEmotion = Label(root, bg="#21556e", fg="#ffffff", text="Predicted emotion :", font=('Shentox', 15))
    root.predictedEmotion.grid(row=3, column=1, padx=10, pady=10)

    root.emotion = Label(root, bg="#21556e", fg="#ffffff", text="undefined", font=('Shentox', 15))
    root.emotion.grid(row=3, column=2, padx=10, pady=10)

    root.accuracyLabel = Label(root, bg="#21556e", fg="#ffffff", text="Accuracy :", font=('Shentox',15))
    root.accuracyLabel.grid(row=5, column=1, padx=10, pady=10)

    root.accuracy = Label(root, bg="#21556e", fg="#ffffff", text="undefined", font=('Shentox',15))
    root.accuracy.grid(row=5, column=2, padx=10, pady=10)

    root.modelLabel = Label(root, bg="#21556e", fg="#ffffff", text="Model :", font=('Shentox',15))
    root.modelLabel.grid(row=4, column=1, padx=10, pady=10)

    root.model = Label(root, bg="#21556e", fg="#ffffff", text="without user adaptation", font=('Shentox',15))
    root.model.grid(row=4, column=2, padx=10, pady=10)

    checkAccuracy()
    ShowFeed()

def ShowFeed():
    ret, frame = root.cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (20,30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255))
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        videoImg = Image.fromarray(cv2image)
        emotion_text = predict_emotion(videoImg)
        root.emotion.config(text = emotion_text)
        imgtk = ImageTk.PhotoImage(image = videoImg)
        root.cameraLabel.configure(image=imgtk)
        root.cameraLabel.imgtk = imgtk
        root.cameraLabel.after(1, ShowFeed)

def get_label(features_file):
    return int(features_file.split('-')[0][-2:])

def load_features(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def checkAccuracy():
    TEST_DIR = 'test_features'
    test_features = os.listdir(TEST_DIR)
    prediction, labels = [], []
    for test_file in test_features:
        features = load_features(os.path.join(TEST_DIR, test_file))
        target_type = "int32"
        label = np.array([[0, 0, 0, 0, 0, 0, 0, 0]]).astype(target_type)
        label[0][get_label(test_file) - 1] = 1
        for i in range(len(features)):
            pred = emotion_predictor.predict(features[i])
            pred_idx = np.argmax(pred.numpy()[0])
            pred_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0]]).astype(target_type)
            pred_label[0][pred_idx] = 1
            prediction.append(pred_label[0])
            labels.append(label[0])
    accuracy = metrics.accuracy_score(labels, prediction)
    root.accuracy.config(text = accuracy)


def changeModel():
    model_file = filedialog.askopenfilename(initialdir="models")
    trained_times = model_file.split('.')[0][-1]
    with open(model_file, 'rb') as f:
        model_params = pickle.load(f)

    emotion_predictor.load_params(model_params)

    if trained_times != 0 :
        root.model.config(text = f"trained with {trained_times} videos per emotion")
    else:
        root.model.config(text = "without user adaptation")
    messagebox.showinfo("SUCCESS", "Model has been changed")
    checkAccuracy()

def setup_models():
    feature_extractor_name = "features_extractor.so"
    vm_feature_extractor = relax.VirtualMachine(tvm.runtime.load_module(feature_extractor_name), tvm.cpu())

    fer_model_name = "fer.so"
    vm_fer = relax.VirtualMachine(tvm.runtime.load_module(fer_model_name), tvm.cpu())

    @I.ir_module
    class FER:
        I.module_attrs({"param_num": 2, "state_num": 0})
        
        @R.function
        def backbone(
            x: R.Tensor((1, 1280), "float32"),
            w0: R.Tensor((1280, 8), "float32"),
            b0: R.Tensor((1, 8), "float32"),
        ):
            with R.dataflow():
                lv0 = R.matmul(x, w0)
                lv1 = R.add(lv0, b0)
                out = R.nn.softmax(lv1, axis = 1)
                R.output(out)
            return out

    pred_sinfo = relax.TensorStructInfo((1, 8), "float32")
    target_sinfo = relax.TensorStructInfo((1, 8), "int32")
    setup_trainer = SetupTrainer(
        CategoricalCrossEntropyLoss(reduction="mean"),
        Adam(0.0001),
        [pred_sinfo, target_sinfo],
    )

    train_mod = setup_trainer(FER)

    dev = tvm.device("cpu", 0)

    trainer = Trainer(train_mod, vm_fer, dev, False)

    with open('models/fer_model_params_trained0.pkl', 'rb') as f:
        model_params = pickle.load(f)

    trainer.load_params(model_params)

    return vm_feature_extractor, trainer


def predict_emotion(img) -> str:
    test_transforms = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        ]
    )

    img_tensor = test_transforms(img)
    inp = tvm.nd.array(img_tensor.numpy().reshape(1, 3, 224, 224)
                        .astype(np.float32))
    features = feature_extractor["main"](inp)
    features = features.asnumpy()
    emotion = emotion_predictor.predict(features)
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    emotion_str = emotions[np.argmax(emotion.numpy()[0])]
    return emotion_str

root = tk.Tk()
root.cap = cv2.VideoCapture(-1)
width, height = 600, 600
root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root.title("Emotion recognition")
root.geometry("700x720")
root.resizable(True, True)
root.configure(background = "#21556e")

feature_extractor, emotion_predictor = setup_models()

createwidgets()
root.mainloop()
