import gradio as gr
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image
import torch
import cv2
import numpy as np
from PIL import Image

is_show_controlnet = True
prompts = ""
neg_prompt = "chinese letter"

repo_id = "calihyper/trad-kor-landscape-black"
#pipe = StableDiffusionPipeline.from_pretrained(repo_id).to("mps")

pipe = StableDiffusionPipeline.from_pretrained(repo_id).to("cuda")

def change_radio(input):
    return input

def output_radio(output):
    print(output)

def predict(prompt, style_prompt, neg_prompt, ins, gs, seed):
    generator = torch.manual_seed(seed)
    global pipe
    output = pipe(
        prompt + style_prompt,
        negative_prompt=neg_prompt,
        generator=generator,
        num_inference_steps=ins,
        guidance_scale=gs
    )
    return output.images[0]

with gr.Blocks() as demo:
    gr.Markdown("# Aiffelthon Choosa Project")

    with gr.Row():
        with gr.Column():
            out_image = gr.Image(shape=(512,512))
            with gr.Column() as diff:
                prompt = gr.Textbox(placeholder="prompts", label="prompt")
                style_prompt = gr.Textbox(placeholder="style prompts", label="style prompt")
                examples = gr.Examples(examples=["<trad-kor-landscape-black>", "<trad-kor-landscape-ink-wash-painting>", "<trad-kor-landscape-thick-brush-strokes>", "<trad-kor-plants-black>", "<trad-kor-plants-color>"], 
                                       inputs=style_prompt,  label="style examples")

                neg_prompt = gr.Textbox(placeholder="negative prompts", value=neg_prompt, label="negative prompt")

                ins = gr.Slider(1, 60, 30, label="inference steps")
                gs = gr.Slider(1, 10, 2.5, step=1, label="guidance scale")

                seed = gr.Slider(0, 10, 2, step=1, label="seed")
    btn1 = gr.Button("실행")
    btn1.click(predict, [ prompt, style_prompt, neg_prompt, ins, gs, seed], out_image)

if __name__ == "__main__":
    demo.launch()    