import torch
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms

import os
import sys
import math
import time
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

module_path = os.path.abspath(os.path.join('D:/PPJ/Model/notebook'))
if module_path not in sys.path:
    sys.path.append(module_path)

from py.vae import NeuralNet as nnet
import py.config as CFG

def load_pretrained_model(encoder_path, decoder_path, device):
    neuralnet = nnet(height=CFG.height, width=CFG.width, channel=CFG.channel, device=CFG.device, ngpu=CFG.ngpu)
    neuralnet.encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    neuralnet.decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))
    return neuralnet

def calculate_heatmap(image, reconstructed_img):
    img_pil = to_pil_image(image.squeeze())
    reconstructed_img_pil = to_pil_image(reconstructed_img.squeeze())

    img_np = np.array(img_pil)
    reconstructed_img_np = np.array(reconstructed_img_pil)

    heatmap = np.zeros_like(img_np, dtype=np.float32)

    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            heatmap[i, j] = math.sqrt(math.pow(img_np[i, j] - reconstructed_img_np[i, j], 2))
    flatten_arr = heatmap.flatten()
    filtered_arr = flatten_arr[flatten_arr > 100]
    var = np.var(filtered_arr)
    return np.array(var).reshape(-1, 1), heatmap, filtered_arr

def plot_heatmap(heatmap, save_path=None):
    plt.imshow(heatmap, cmap='Reds')
    plt.colorbar()
    plt.title("Heatmap of Pixel Intensity Differences")
    if save_path:
        plt.savefig(save_path)
    plt.cla()
    plt.clf()
    plt.close()

def plot_histogram(filtered_arr, save_path=None):
    plt.hist(filtered_arr.ravel(), bins=256, range=(0, 255), fc='k', ec='k')
    plt.title("Histogram of Heatmap Values")
    plt.xlabel("Pixel Intensity Difference")
    plt.ylabel("Frequency")
    if save_path:
        plt.savefig(save_path)
    plt.cla()
    plt.clf()
    plt.close()

def save_result(text, file_path):
    with open(file_path, 'w') as file:
        file.write(text)

def main():
    vae = load_pretrained_model(r'D:\PPJ\Model\result\encoder_epoch_30.pth', r'D:\PPJ\Model\result\decoder_epoch_30.pth', CFG.device)
    vae.encoder.eval()
    vae.decoder.eval()

    transform = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize((0.0), (1.0))
])

    
    print('프로그램 준비 완료')

    while(True):
        if os.path.exists(CFG.image_path):
            image = Image.open(CFG.image_path)
            image = CFG.transform(image).to(CFG.device).unsqueeze(1)
            with torch.no_grad():
                conv_out = vae.encoder(image)
                reconstructed_img = vae.decoder(conv_out)
            
            pred_var, heatmap, filtered_arr = calculate_heatmap(image, reconstructed_img)
            plot_heatmap(heatmap, save_path=r'\\172.28.1.207\SharedDIR\Predict_result\heatmap.png')
            plot_histogram(filtered_arr, save_path=r'\\172.28.1.207\SharedDIR\Predict_result\histogram.png')

            result_text = f'result : Good, var : {pred_var}' if pred_var <= CFG.threshold else f'result : Bad, var : {pred_var}'
            save_result(result_text, r'\\172.28.1.207\SharedDIR\Predict_result\result.txt')

            os.remove(CFG.image_path)
            print(f"{CFG.image_path} 파일을 삭제했습니다.")
        else:
            print(f"{CFG.image_path} 파일을 기다리는 중입니다.")
            time.sleep(5)
            pass

if __name__ == "__main__":
    main()