import gradio as gr
from load_image import load_img
import spaces
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np

torch.set_float32_matmul_precision(["high", "highest"][0])

# load 2 models

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)


# RMBG2 = AutoModelForImageSegmentation.from_pretrained(
#     "briaai/RMBG-2.0", trust_remote_code=True
# )

# Keep them in a dict to switch easily
models_dict = {
    "BiRefNet": birefnet,
    # "RMBG-2.0": RMBG2
}

# Transform

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

@spaces.GPU
def process(image: Image.Image, model_choice: str):
    """
    Runs inference to remove the background (adds alpha) 
    with the chosen segmentation model.
    """
    # Select the model
    current_model = models_dict[model_choice]

    # Prepare image
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        # Each model returns a list of preds in its forward, 
        # so we take the last element, apply sigmoid, and move to CPU
        preds = current_model(input_images)[-1].sigmoid().cpu()

    # Convert single-channel pred to a PIL mask
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)

    # Resize the mask back to original image size
    mask = pred_pil.resize(image_size)

    # Add alpha channel to the original
    image.putalpha(mask)
    return image

def fn(source: str, model_choice: str):
    """
    Used by Tab 1 & Tab 2 to produce a processed image with alpha.
    - 'source' is either a file path (type="filepath") or 
      a URL string (textbox).
    - 'model_choice' is the user's selection from the radio.
    """
    # Load from local path or URL
    im = load_img(source, output_type="pil")
    im = im.convert("RGB")

    # Process
    processed_image = process(im, model_choice)
    return processed_image

def process_file(file_path: str, model_choice: str):
    """
    For Tab 3 (file output).
    - Accepts a local path, returns path to a new .png with alpha channel.
    - 'model_choice' is also passed in for selecting the model.
    """
    name_path = file_path.rsplit(".", 1)[0] + ".png"
    im = load_img(file_path, output_type="pil")
    im = im.convert("RGB")

    # Run the chosen model
    transparent = process(im, model_choice)
    transparent.save(name_path)
    return name_path


# GRadio UI

# model_selector_1 = gr.Radio(
#     choices=["BiRefNet","RMBG-2.0"],
#     value="BiRefNet",
#     label="Select Model"
# )
# model_selector_2 = gr.Radio(
#     choices=["BiRefNet","RMBG-2.0"],
#     value="BiRefNet",
#     label="Select Model"
# )
# model_selector_3 = gr.Radio(
#     choices=["BiRefNet", "RMBG-2.0"],
#     value="BiRefNet",
#     label="Select Model"
# )

radio_opts = ["BiRefNet"]            # single choice everywhere

model_selector_1 = gr.Radio(radio_opts, value="BiRefNet", label="Select Model")
model_selector_2 = gr.Radio(radio_opts, value="BiRefNet", label="Select Model")
model_selector_3 = gr.Radio(radio_opts, value="BiRefNet", label="Select Model")

# Outputs for tabs 1 & 2: single processed image
processed_img_upload = gr.Image(label="Processed Image (Upload)", type="pil")
processed_img_url = gr.Image(label="Processed Image (URL)", type="pil")

# For uploading local files
image_upload = gr.Image(label="Upload an image", type="filepath")
image_file_upload = gr.Image(label="Upload an image", type="filepath")

# For Tab 2 (URL input)
url_input = gr.Textbox(label="Paste an image URL")

# For Tab 3 (file output)
output_file = gr.File(label="Output PNG File")

# Tab 1: local image -> processed image
tab1 = gr.Interface(
    fn=fn,
    inputs=[image_upload, model_selector_1],
    outputs=processed_img_upload,
    api_name="image",
    description="Upload an image and choose your background removal model."
)

# Tab 2: URL input -> processed image
tab2 = gr.Interface(
    fn=fn,
    inputs=[url_input, model_selector_2],
    outputs=processed_img_url,
    api_name="text",
    description="Paste an image URL and choose your background removal model."
)

# Tab 3: file output -> returns path to .png
tab3 = gr.Interface(
    fn=process_file,
    inputs=[image_file_upload, model_selector_3],
    outputs=output_file,
    api_name="png",
    description="Upload an image, choose a model, and get a transparent PNG."
)

# Combine all tabs
demo = gr.TabbedInterface(
    [tab1, tab2, tab3],
    ["Image Upload", "URL Input", "File Output"],
    title="Background Removal Tool"
)

if __name__ == "__main__":
    demo.launch(show_error=True, share=True)
