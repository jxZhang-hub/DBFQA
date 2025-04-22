import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import subprocess
import torch
from models.DBFQA import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_edge_points(ps_path, height, width):
    try:
        with open(ps_path, "r") as f:
            cords = []
            count = 0
            for line in f:
                line = line.strip().split(" ")
                if line[-1] == 'edge':
                    if count == 0:
                        count = 1
                    else:
                        cords.append((int(line[0]), int(line[1])))
        out = np.full((height, width), 255, dtype=int)
        for x, y in cords:
            if x < width and y < height:
                out[y, x] = 0
        return out
    except Exception as e:
        messagebox.showerror("Error", f"Error reading edge points: {str(e)}")
        return None

def perform_edge_detection(image_path, use_ll):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to read the image.")
        height, width = image.shape[:2]
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if not use_ll:
            edges = cv2.Canny(gray_image, 100, 200)
            _, binary_edge_image = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
        else:
            args = ["./pgmloglin.exe", "-E", image_path, "./ps.txt"]
            subprocess.run(args, check=True)
            edges = read_edge_points("./ps.txt", height, width)
            if edges is not None:
                edges = np.flipud(edges)
                binary_edge_image = np.rot90(edges, 3)
            else:
                return None
        return binary_edge_image
    except Exception as e:
        messagebox.showerror("Error", f"Edge detection error: {str(e)}")
        return None

def preprocess_image(image_path, use_ll):
    try:
        img = Image.open(image_path).convert('L')
        ret1 = np.array(img) / 255.0
        ret1 = cv2.resize(ret1, (224, 224))
        imge = perform_edge_detection(image_path, use_ll)
        if imge is not None:
            ret2 = np.array(imge) / 255.0
            ret2 = cv2.resize(ret2, (224, 224))
            return torch.tensor(ret1).unsqueeze(0), torch.tensor(ret2).unsqueeze(0)
        return None, None
    except Exception as e:
        messagebox.showerror("Error", f"Image preprocessing error: {str(e)}")
        return None, None

def calculate_image_quality(image_path, is_ll_checked):
    try:
        x1, x2 = preprocess_image(image_path, is_ll_checked)
        if x1 is not None and x2 is not None:
            x1 = x1.to(device)
            x2 = x2.to(device)
            score = model(x1, x2)
            return int(score * 100)
        return None
    except Exception as e:
        messagebox.showerror("Error", f"Quality calculation error: {str(e)}")
        return None

def show_image(image_path, image_label):
    try:
        img = Image.open(image_path)
        img.thumbnail((200, 200))
        photo = ImageTk.PhotoImage(img)
        image_label.config(image=photo)
        image_label.image = photo
    except Exception as e:
        messagebox.showerror("Error", f"Error displaying image: {str(e)}")

def browse_image(entry_path, image_label):
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.psd;*.tif;*.tiff")])
    if file_path:
        entry_path.delete(0, tk.END)
        entry_path.insert(0, file_path)
        show_image(file_path, image_label)

def display_score(entry_path, result_label, is_ll_checked):
    image_path = entry_path.get()
    if not image_path:
        messagebox.showerror("Error", "Please enter a valid image path.")
        return
    score = calculate_image_quality(image_path, is_ll_checked)
    if score is not None:
        result_label.config(text=f"Quality score: {score}")

try:
    model = Net(batch_size=1).to(device)
    model.load_state_dict(torch.load("./models/checkpoints/model.pth", weights_only=True))
    model.eval()
except Exception as e:
    messagebox.showerror("Error", f"Failed to load the model: {str(e)}")
    raise

#Main window
root = tk.Tk()
root.title("DB-FQA demo")
root.geometry("300x500")
root.resizable(False, False)

#Image path and button
label_path = tk.Label(root, text="Image path:")
label_path.pack(pady=10)

entry_path = tk.Entry(root, width=20)
entry_path.pack(pady=5)

button_browse = tk.Button(root, text="Browse", command=lambda: browse_image(entry_path, image_label))
button_browse.pack(pady=5)

#Tick LL
ll_var = tk.BooleanVar()
checkbutton_ll = tk.Checkbutton(root, text="LL", variable=ll_var)
checkbutton_ll.pack(pady=5)

button_calculate = tk.Button(root, text="Quality assessment",
                             command=lambda: display_score(entry_path, result_label, ll_var.get()))
button_calculate.pack(pady=20)

#Place where image is
image_label = tk.Label(root)
image_label.pack(pady=10)

#Initially a blank image
blank_img = Image.new("RGB", (200, 200), (255, 255, 255))
photo = ImageTk.PhotoImage(blank_img)
image_label.config(image=photo)
image_label.image = photo

#showing result
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

#run
root.mainloop()