# Introduction
choosa is a model that fine-tuned the [StableDiffsuion_v1-5 model](https://github.com/runwayml/stable-diffusion) with the [Textual Inversion](https://huggingface.co/docs/diffusers/training/text_inversion) method for Korean traditional paintings.

# environment
**Installing the dependencies**    
```python
pip install .  # first
pip install -r requirements.txt # Second
```

# inference
**1. Text to Image**   
You can find more detailed information [Text-to-Image Generation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)   
```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "calihyper/trad-kor-landscape-black"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "a painting of Eiffel tower in Paris <trad-kor-landscape-ink-wash-painting>"
negative_prompt = "chinese writing"
image = pipe(prompt, negative_prompt = negative_prompt, num_inference_steps=20, guidance_scale=2.5).images[0]

image.save("Eiffel tower.png")
```

**2. Controlnet-canny**   
You can find more detailed information [Text-to-Image Generation with ControlNet Conditioning](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/controlnet)

```python
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
# Let's load your image
image = load_image(
    "your image.png"
)
```
First, we need to install opencv:
```python
pip install opencv-contrib-python
```
Next, let’s also install all required Hugging Face libraries:
```python
pip install diffusers transformers git+https://github.com/huggingface/accelerate.git
```
Now, we load our finetuned Model as well as the ControlNet for canny edges.

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

model_id = "calihyper/trad-kor-landscape-black"
controlnet = ControlNetModel.from_pretrained("calihyper/trad-kor-controlnet", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16
)
```
To speed-up things and reduce memory, let’s enable model offloading and use the fast UniPCMultistepScheduler.

```python
from diffusers import UniPCMultistepScheduler

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# this command loads the individual model components on GPU on-demand.
pipe.enable_model_cpu_offload()
```
Finally, we can run the pipeline:

```python
generator = torch.manual_seed(0)
prompt = ""
negative_prompt = ""
out_image = pipe(
    prompt=prompt, negative_prompt = negative_prompt, num_inference_steps=20, guidance_scale = 2.5, generator=generator, image=canny_image
).images[0]
```
# examples


# License



# Reference
1. https://arxiv.org/abs/2302.05543
2. https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines
3. https://huggingface.co/blog/controlnet
4. https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img
5. https://huggingface.co/docs/diffusers/training/text_inversion
6. https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion
