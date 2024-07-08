[**_Model 제작 과정 및 결과_**](https://github.com/MBV-and-Kids/Model/blob/main/model.md)

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
