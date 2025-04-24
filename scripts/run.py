import os
import argparse

model_ckpt = 'models/model.ckpt'
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="Beginning with a beach scene, the camera gradually draws in closer as waves lap against the reef. Then the camera slowly pans right and a large area of sea is revealed")
parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
os.makedirs('prompts', exist_ok=True)

import cv2
import numpy as np
import shutil
from pathlib import Path
from scripts.llm import MyLLM
import datetime
import imageio
from PIL import Image

prompt = args.prompt

my_llm = MyLLM()
try:
    question = my_llm.gen_question(prompt)
    outputs = my_llm.ask_question(question)
    outputs = list(eval(outputs))
except:
    print("Error")
    exit(0)

# outputs = ["field and blue sky, house in the distance", "Zoom In", "large fields", "Pan Left"]

action_dict = {
    "Zoom In": "models/MotionLoRA/v2_lora_ZoomIn.ckpt",
    "Zoom Out": "models/MotionLoRA/v2_lora_ZoomOut.ckpt",
    "Pan Left": "models/MotionLoRA/v2_lora_PanLeft.ckpt",
    "Pan Right": "models/MotionLoRA/v2_lora_PanRight.ckpt",
    "Tilt Up": "models/MotionLoRA/v2_lora_TiltUp.ckpt",
    "Tilt Down": "models/MotionLoRA/v2_lora_TiltDown.ckpt"
}
default_action = "Zoom In"

prompt_format = '''- inference_config: "configs/inference/inference-v2.yaml"
  motion_module:    "models/Motion_Module/mm_sd_v15_v2.ckpt"

  dreambooth_path: "models/DreamBooth_LoRA/realisticVisionV51_v51VAE.safetensors"
  lora_model_path: ""

  seed:          {}
  steps:          25
  guidance_scale: 9

  prompt:
  - {}, high quality, film grain, Fujifilm XT3
  n_prompt:
  - semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, moving arms, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long
'''

controlnet_format = '''
  controlnet_config: "configs/inference/sparsectrl/condition.yaml"
  controlnet_path:   "{}"

  controlnet_image_indexs: [0]
  controlnet_images:
    - "{}"
'''

motion_lora_format = '''
  motion_module_lora_configs:
    - path:  "{}"
      alpha: 1.0
'''

last_save_dir = None
for i in range(0, len(outputs), 2):
  try:
    action = action_dict[outputs[i + 1]]
  except:
    action = default_action
  with open('prompts/configs/{}.yaml'.format(int(i / 2)), 'w') as f:
    f.write(prompt_format.format(-1, outputs[i]))
    if i + 1 < len(outputs):
      f.write(motion_lora_format.format(action))
    if i > 0:
      f.write(controlnet_format.format(model_ckpt, os.path.join(last_save_dir, "sample.gif_end.png")))
    
  time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
  savedir = "samples/{}-{}".format(int(i / 2), time_str)
  cmd = "python -m scripts.animate --config prompts/configs/{}.yaml --savedir {} --gpu_id {} --index {}".format(int(i / 2), savedir, args.gpu_id, int(i / 2))
  print(cmd)
  os.system(cmd)
  last_save_dir = savedir
  shutil.copy(os.path.join(savedir, "sample.gif"), "prompts/outputs/{}.gif".format(int(i / 2)))


file_names = sorted(list(os.listdir('prompts/outputs')))
os.makedirs('prompts/samples', exist_ok=True)
j = 0
for index, file_name in enumerate(file_names):
    img = Image.open(f'prompts/outputs/{file_name}')
    i = 0
    while True:
        try:
            img.seek(i)
            img.convert('RGB').save(f'prompts/samples/{j}.png')
            j += 1
            i += 1
        except Exception as e:
            print(str(e))
            break
tot = j

def adjust_brightness_and_saturation(image_ref, image_target):
    """
    Adjust the brightness and saturation of the target image to match the reference image.

    Parameters:
        image_ref (numpy.ndarray): Reference image.
        image_target (numpy.ndarray): Target image.

    Returns:
        numpy.ndarray: Aligned target image.
    """
    # Convert images to LAB and HSV color spaces
    lab_image_ref = cv2.cvtColor(image_ref, cv2.COLOR_BGR2LAB)
    lab_image_target = cv2.cvtColor(image_target, cv2.COLOR_BGR2LAB)
    hsv_image_ref = cv2.cvtColor(image_ref, cv2.COLOR_BGR2HSV)
    hsv_image_target = cv2.cvtColor(image_target, cv2.COLOR_BGR2HSV)

    # Compute mean brightness and saturation of both images
    mean_brightness_ref = np.mean(lab_image_ref[:, :, 0])
    mean_brightness_target = np.mean(lab_image_target[:, :, 0])
    mean_saturation_ref = np.mean(hsv_image_ref[:, :, 1])
    mean_saturation_target = np.mean(hsv_image_target[:, :, 1])

    # Compute adjustment factors for brightness and saturation
    brightness_factor = mean_brightness_ref - mean_brightness_target
    saturation_factor = mean_saturation_ref / mean_saturation_target * 1.15

    # Adjust brightness of target image
    lab_image_target[:, :, 0] = np.clip(lab_image_target[:, :, 0] + brightness_factor, 0, 255)

    # Adjust saturation of target image
    hsv_image_target[:, :, 1] = np.clip(hsv_image_target[:, :, 1] * saturation_factor, 0, 255)

    # Convert adjusted LAB and HSV images back to BGR color space
    adjusted_lab_image = cv2.cvtColor(lab_image_target, cv2.COLOR_LAB2BGR)
    adjusted_image = cv2.cvtColor(hsv_image_target, cv2.COLOR_HSV2BGR)

    return adjusted_image
  

def adjust(ref, tar, add_on=0, mul_fac=1.0):
    reference_image = cv2.imread("prompts/samples/" + imagelist[ref])
    target_image = cv2.imread("prompts/samples/" + imagelist[tar])
    adjusted_image = adjust_brightness_and_saturation(reference_image, target_image)
    cv2.imwrite("prompts/samples/" + imagelist[tar], adjusted_image)


img_list = []
imagelist = os.listdir('prompts/samples')
imagelist.sort(key=lambda x: int(x.split('.')[0]))
for i in range(0, len(imagelist), 16):
    for j in range(0, 16, 1):
        if j == 15:
            continue
        index = i + j
        if index >= tot:
            continue
        if i > 0:
            adjust(15, index, add_on=20, mul_fac=0.6)
        img_list.append(imageio.v2.imread('prompts/samples/' + imagelist[index]))
imageio.mimsave('prompts/final.gif', img_list, fps=8)