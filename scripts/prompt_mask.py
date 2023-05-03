from transformers import AutoProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch
import argparse
import cv2
import os
import numpy as np

from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images, images, fix_seed
from modules.shared import opts, cmd_opts, state
import modules.scripts as scripts
import gradio as gr
from fastapi import FastAPI, Body

def import_or_install(package,pip_name=None):
    import importlib
    import subprocess
    if pip_name is None:
        pip_name=package
    try:
        importlib.import_module(package)
        print(f"{package} is already installed")
    except ImportError:
        print(f"{package} is not installed, installing now...")
        subprocess.call(['pip', 'install', package])
        print(f"{package} has been installed")
__version__ = "0.0.1"
processor= model=device=None
def initialize():
    device="cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./models/CIDAS"
    if not os.path.exists(model_path):
        processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        processor.save_pretrained(model_path)
        model.save_pretrained(model_path)
    else:
        processor = AutoProcessor.from_pretrained(model_path)
        model = CLIPSegForImageSegmentation.from_pretrained(model_path)
    model.to(device)
    return processor, model,device

def resize_img(img, w, h):
    if img.shape[0] + img.shape[1] < h + w:
        interpolation = interpolation=cv2.INTER_CUBIC
    else:
        interpolation = interpolation=cv2.INTER_AREA

    return cv2.resize(img, (w, h), interpolation=interpolation)

def create_mask(image,clipseg_mask_prompt,clipseg_exclude_prompt,only_mask,clipseg_mask_threshold=0.4,mask_blur_size=11,mask_blur_size2=11):
    import_or_install("rembg","rembg[gpu]")
    from rembg import remove
    global processor, model,device
    if model is None:
        processor, model,device=initialize()
    texts = [x.strip() for x in clipseg_mask_prompt.split(',')]
    exclude_texts = [x.strip() for x in clipseg_exclude_prompt.split(',')] if clipseg_exclude_prompt else None
    
    if exclude_texts:
        all_texts = texts + exclude_texts
    else:
        all_texts = texts
    
    inputs = processor(text=all_texts, images=[image] * len(all_texts), padding="max_length", return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    if len(all_texts) == 1:
        preds = outputs.logits.unsqueeze(0)
    else:
        preds = outputs.logits

    mask_img = None

    for i in range(len(all_texts)):
        x = torch.sigmoid(preds[i])
        x = x.to('cpu').detach().numpy()

#            x[x < clipseg_mask_threshold] = 0
        x = x > clipseg_mask_threshold

        if i < len(texts):
            if mask_img is None:
                mask_img = x
            else:
                mask_img = np.maximum(mask_img,x)
        else:
            mask_img[x > 0] = 0

    mask_img = mask_img*255
    mask_img = mask_img.astype(np.uint8)
    
    if mask_blur_size > 0:
        mask_blur_size = mask_blur_size//2 * 2 + 1
        mask_img = cv2.medianBlur(mask_img, mask_blur_size)

    if mask_blur_size2 > 0:
        mask_blur_size2 = mask_blur_size2//2 * 2 + 1
        mask_img = cv2.GaussianBlur(mask_img, (mask_blur_size2, mask_blur_size2), 0)

    mask_img = resize_img(mask_img, image.width, image.height)

    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
    if only_mask:
        return Image.fromarray(mask_img)
    else:
        from PIL import ImageOps
        np_img = np.array(image)
        mask_img = Image.fromarray(mask_img)
        mask_img=ImageOps.invert(mask_img)
        mask_img = np.array(mask_img)
        sub_img=cv2.subtract(np_img, mask_img)
        sub_img = Image.fromarray(sub_img)
        transparent_image=remove(sub_img)
        return transparent_image

class Script(scripts.Script):
    def title(self):
        return "Prompt Mask"
    def show(self, is_img2img):
        return is_img2img
    def ui(self, is_img2img):
        if not is_img2img: return
        prompts=gr.inputs.Textbox(lines=2, label="Prompts", default="people")
        neg_prompts=gr.inputs.Textbox(lines=2, label="Negative Prompts", default="")
        threshold = gr.inputs.Slider(minimum=0, maximum=1, step=0.01, label='Threshold', default=0.4)
        only_mask=gr.inputs.Checkbox(label="Only Mask", default=False)
        save_mask=gr.inputs.Checkbox(label="Save Mask", default=True)
        mask_blur_median=gr.inputs.Slider(minimum=0, maximum=100, step=1, label='Mask Blur Median', default=11)
        mask_blur_gaussian=gr.inputs.Slider(minimum=0, maximum=100, step=1, label='Mask Blur Gaussian', default=11)
        with gr.Blocks() as demo:
            with gr.Row().style(equal_height=True):
                image=gr.Image(type="pil")
                mask=gr.Image(type="pil")
        btn = gr.Button(value="Preview Remove Background")
        if image is not None:
            btn.click(create_mask, inputs=[image,prompts,neg_prompts,only_mask,threshold,mask_blur_median,mask_blur_gaussian],\
                       outputs=[mask])
        return [prompts, neg_prompts,only_mask,threshold, save_mask, mask_blur_median, mask_blur_gaussian]
    def run(self,p, prompts, neg_prompts,only_mask,threshold, save_mask, mask_blur_median, mask_blur_gaussian):
        mask=create_mask(p.init_images[0],prompts,neg_prompts,True,threshold,mask_blur_median,mask_blur_gaussian)
        if save_mask:
            images.save_image(mask, p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, p=p)
        p.image_mask=mask
        proc = process_images(p)
        proc.images.append(mask)
        return proc
    
def prompt_mask_api(_: gr.Blocks, app: FastAPI):
    @app.get("/prompt_mask/status")
    async def get_status():
        return {"status": "ok", "version": "1.0.0"}
    @app.post("/prompt_mask/remove-background")
    async def post_create_mask(image_str: str = Body(...), prompts: str = Body(...), neg_prompts: str = Body(...,embed=True),\
                               only_mask: bool = Body(...), threshold: float = Body(...),\
                               mask_blur_median: int = Body(...), mask_blur_gaussian: int = Body(...)):
        import modules.shared as shared
        if shared.tecky_auth.demo_available() is False:
            return {"error": "Server Busy"}
        import base64
        import io
        image_bytes = base64.b64decode(image_str)
        image = Image.open(io.BytesIO(image_bytes),formats=['PNG']).convert('RGB')
        mask=create_mask(image,prompts,neg_prompts,only_mask,threshold,mask_blur_median,mask_blur_gaussian)
        buffered = io.BytesIO()
        mask.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        return {"mask": img_str}
try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(prompt_mask_api)
except:
    pass