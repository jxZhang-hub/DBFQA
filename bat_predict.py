import tqdm
from models.DBFQA import Net
import cv2 as cv
import os
from PIL import Image
import torch
import numpy as np
import cv2
import subprocess
import argparse

def findps(ps_path, height, width):
    #turning .ps file into edge map
    f = open(ps_path, "r")

    #the cordinate of edge points
    cords = []
    count = 0
    for line in f:
        line = line.strip()
        line = line.split(" ")
        if line[-1] == 'edge' and count == 0:
            count = 1
        elif line[-1] == 'edge' and count != 0:
            cords.append((line[0], line[1]))
    f.close()

    #drawing edge map
    out = np.zeros((height, width), dtype=int)
    for i in range(height):
        for j in range(width):
            out[i, j] = 255
    for cord in cords:
        x, y = cord
        x = int(x)
        y = int(y)
        if x < width or y < height:
            continue
        else:
            out[x, y] = 0

    return out
def edge_detection_and_binary(image_path, LL):

    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    #getting the grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if LL == False:
        edges = cv2.Canny(gray_image, 100, 200)
        _, binary_edge_image = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    else:
        args = ["./pgmloglin.exe", "-E", image_path, "./ps.txt"]
        #subprocess.run(args, stdout=subprocess.DEVNULL)
        subprocess.run(args)
        edges = findps("./ps.txt", height, width)
        edges = np.flipud(edges)
        binary_edge_image = np.rot90(edges, 3)

    return binary_edge_image

def preprocess(path, LL):
    img = Image.open(path)
    img = img.convert('L')
    ret1 = img.copy()
    img.close()
    ret1 = np.array(ret1) / 255.0
    ret1 = cv.resize(ret1, (224, 224))

    imge = edge_detection_and_binary(path, LL=LL)
    ret2 = imge.copy()
    ret2 = np.array(ret2) / 255.0
    ret2 = cv.resize(ret2, (224, 224))
    return torch.tensor(ret1).unsqueeze(0), torch.tensor(ret2).unsqueeze(0)

def prediction(path_dir, output_path, LL, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_names = os.listdir(path_dir)
    x1 = []
    x2 = []
    for img_name in img_names:
        x1_, x2_ = preprocess(path_dir + img_name, LL)
        x1.append(x1_.to(device))
        x2.append(x2_.to(device))


    model = Net(batch_size=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    pred = []
    with tqdm.tqdm(total=len(img_names)) as pbar:
        for i in range(len(x1)):
            score = model(x1[i], x2[i])
            pred.append(score.item())
            pbar.update(1)
    with open(output_path, "w") as f:
        for i in range(len(x1)):
            f.write(str(img_names[i] + "\t" + str(int(100 * pred[i]))) + "\n")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DB-FQA Fingerprint quality assessment')
    parser.add_argument('--i_dir', type=str, default='./input_images/', help='Directory containing input images')
    parser.add_argument('--o_file', type=str, default='./output_scores/score.txt', help='Output file to save scores')
    parser.add_argument("--LL", type=bool, default=False, help="Whether to use L/L operator")
    parser.add_argument('--ckpt', type=str, default='./models/checkpoints/model.pth', help='Checkpoint path')

    args = parser.parse_args()
    prediction(args.i_dir, args.o_file, args.LL, args.ckpt)