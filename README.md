
# 🖼️ Background Removal Tool (Gradio UI + BiRefNet)

**Clean, Simple, Fast Background Removal with BiRefNet!**  
Built using 🤗 Transformers, 🧠 PyTorch, and 🌐 Gradio. No Photoshop? No problem.

## 🚀 Features

- 🧠 Powered by [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) — a robust transformer-based image segmentation model.
- 🎛️ Easy-to-use **Gradio** web interface with 3 different modes:
  - 📁 Upload image
  - 🌐 Paste URL
  - 💾 Download transparent `.png`
- ⚙️ Built-in GPU support for faster inference on Hugging Face Spaces or local runtime.
- 🧼 Returns high-quality transparent background images (RGBA).

---

## 📸 Demo

Try the live demo (hosted via Hugging Face Spaces):  
👉 [Click here to try it out](https://huggingface.co/spaces/shemayons/BACKGROUND-REMOVAL)


---

## 🔍 Sample Output

| Input Image | Output (Transparent BG) |
|-------------|--------------------------|
| ![input](input.png) | ![output](output.png) |

---
## 🧪 How it Works

> We load your image, transform it to the model's input format, and use BiRefNet to predict a segmentation mask. The mask is then applied as an **alpha channel** for background transparency.

### Model Pipeline

```text
Image → Preprocessing → BiRefNet → Foreground Mask → Add Alpha Channel → Transparent PNG
```

---

## 🛠️ Installation

```bash
git clone repository
cd repository-name

pip install -r requirements.txt
```

> **Requirements:**
> - Python 3.8+
> - PyTorch
> - Transformers
> - Gradio
> - Pillow
> - Torchvision

---

## 💻 Run the App

```bash
python app.py
```

Once launched, it will open a Gradio interface with three tabs:
- **Image Upload**
- **URL Input**
- **File Output**

---

## 📂 Project Structure

```
├── app.py                  # Main Gradio application
├── load_image.py           # Loads image from local or URL
├── requirements.txt
└── README.md
```



## 🧠 Model Info

- **BiRefNet** from [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)
- Segmentation-based background removal
- Highly accurate with clean edge detection and mask generation

---

## 🔧 TODO

- [ ] Add support for [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
- [ ] Batch processing
- [ ] REST API endpoint (FastAPI)
- [ ] Upload to Hugging Face Space (public)

---

## 🧑‍💻 Author

Made with ❤️ by [Shemayon Soloman](https://www.linkedin.com/in/shemayon-soloman-b32387218/)

---

## 📜 License

This project is licensed under the MIT License.  
Feel free to fork and remix!

---

## ⭐ Show your support

If you like this project, give it a ⭐ on GitHub and try the [demo](https://huggingface.co/spaces/shemayons/BACKGROUND-REMOVAL)!

