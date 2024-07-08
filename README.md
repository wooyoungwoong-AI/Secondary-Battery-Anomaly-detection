# Data
---
### Data Augmentation
#### 이 프로젝트에서는 2차 전지 배터리 캡의 이미지를 4가지 각도로 회전시켰습니다.
* 0도 (원본 이미지)
* 90도
* 180도
* 270

#### 가우시안 노이즈 추가
이미지의 각 회전된 버전에 가우시안 노이즈를 적용하여 데이터의 다양성을 증가시킵니다.

* 각 증강 방법 및 이미지 정규화에 대해서는 여기서 볼 수 있습니다. [**_Custom Dataset_**](https://github.com/MBV-and-Kids/Model/blob/main/notebook/CNNCustomDataset.ipynb)
* GAN을 이용하여 Generate 한 코드는 여기서 볼 수 있습니다. [**_GAN_**](https://github.com/MBV-and-Kids/Model/blob/main/notebook/GAN/GANCreateGood.ipynb)
---

### Data Statistics

<img width="364" alt="image_cnt_graph" src="https://github.com/MBV-and-Kids/Model/assets/136695011/763484b1-f423-4f96-96c3-8d3d5e0a099a">

---

make by [**_JKpon_**](https://github.com/orgs/MBV-and-Kids/people/LeeJeGg)

---
# Model
---
### Using AutoEncoder
#### 데이터셋의 특성 상 Normal Class의 Data와 Error Class Data의 분포가 크다 보니 Anomaly dectection을 하는 방법을 채택하였습니다.
![auto_encoder](https://github.com/wooyoungwoong-AI/wooyoungwoong-AI/assets/136695011/4c7a4723-0314-4a20-8bf8-c0e157de830b)

1. AutoEncoder의 backbone은 여기서 보실 수 있습니다. [**_Backbone_**](https://github.com/MBV-and-Kids/Model/blob/main/notebook/py/vae.py)
2. AutoEncoder에 Normal Class Data 만 학습 시킵니다. 학습 로직은 여기서 볼 수 있습니다. [**_Training logic_**](https://github.com/MBV-and-Kids/Model/blob/main/notebook/py/Training.py)
3. 학습 후 roc 그래프와 heatmap 등을 그려보며 학습 결과를 확인하고 그에 따른 hyperparameter를 바꾸어주었습니다. 실행 파일은 여기서 볼 수 있습니다. [**_Run_**](https://github.com/MBV-and-Kids/Model/blob/main/notebook/run_colab_ver.ipynb)
4. 학습 완료 후 학습 한 모델을 토대로 추론 과정을 진행 하였습니다. 추론 로직은 여기서 볼 수 있습니다. [**_Predict Logic_**](https://github.com/MBV-and-Kids/Model/blob/main/notebook/py/predict.py)
5. 새부적인 각 조건에 대한 코드는 여기서 보실 수 있으십니다. [**_Config_**](https://github.com/MBV-and-Kids/Model/blob/main/notebook/py/config.py)
---

### Heatmap

<img width="333" alt="heatmap" src="https://github.com/wooyoungwoong-AI/wooyoungwoong-AI/assets/136695011/a5accc95-4739-4c35-bf63-b9ce1880daaa">

---

### Histogram

<img width="359" alt="histogram" src="https://github.com/wooyoungwoong-AI/wooyoungwoong-AI/assets/136695011/de209bc4-ef5d-4550-8a49-c966caba047a">

---

### ROC

<img width="364" alt="roc" src="https://github.com/wooyoungwoong-AI/wooyoungwoong-AI/assets/136695011/c9b1a574-1038-460f-ac34-7fa2cd34c52d">

---

make by [**_YoungWoong_**](https://github.com/wooyoungwoong-AI)
