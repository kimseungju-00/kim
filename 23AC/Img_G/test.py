from tqdm import tqdm
import numpy as np
import hashlib
import keras_cv
from tensorflow import keras
import PIL
import os

class_images_dir = "dog_leash_human"
os.makedirs(class_images_dir, exist_ok=True)

keras.mixed_precision.set_global_policy("float32")

model = keras_cv.models.StableDiffusionV2(img_height=512, img_width=512, jit_compile=True)

class_prompt = ("A photo of a person walking with a dog in the park") #as seen from the side"
                #"The person holds the dog's leash in one hand"
                #"The photo should encompass the full side view of both the person and the dog"
                #"The background features a park or a walking path, capturing a complete scene where both the person and the dog are fully visible from the side.")
num_imgs_to_generate = 10
for i in tqdm(range(num_imgs_to_generate)):
    images = model.text_to_image(
        class_prompt,
        batch_size=1,          # default = 1
        num_steps=100,          # default = 50
        unconditional_guidance_scale=9.5 # default = 7.5
    )
    idx = np.random.choice(len(images))
    selected_image = PIL.Image.fromarray(images[idx])
    hash_image = hashlib.sha1(selected_image.tobytes()).hexdigest()
    image_filename = os.path.join(class_images_dir, f"{hash_image}.jpg")
    selected_image.save(image_filename)