from tqdm import tqdm
import numpy as np
import hashlib
import keras_cv
from tensorflow import keras
import PIL
import os

class_images_dir = "dog_human_08"
os.makedirs(class_images_dir, exist_ok=True)

keras.mixed_precision.set_global_policy("float32")

model = keras_cv.models.StableDiffusionV2(img_height=512, img_width=512, jit_compile=True)

class_prompt = "((best quality)), ((high quality)), ((a realistic photograph)), (((a person take a walk in the park with a dog))), (((no leash))), taken from a distance of 5 meters, ((8K)), ((UHD))"
num_imgs_to_generate = 3000
for i in tqdm(range(num_imgs_to_generate)):
    images = model.text_to_image(
        class_prompt,
        negative_prompt = "((ugly)), ((tiling)), ((poorly drawn face)), ((out of frame)), ((extra limbs)), ((body out of frame)), bad anatomy, ((watermark)), signature, ((cut off)), low contrast, ((bad art)), distorted face",
        batch_size=3,
        #unconditional_guidance_scale=9.5
    )
    idx = np.random.choice(len(images))
    selected_image = PIL.Image.fromarray(images[idx])
    hash_image = hashlib.sha1(selected_image.tobytes()).hexdigest()
    image_filename = os.path.join(class_images_dir, f"{hash_image}.jpg")
    selected_image.save(image_filename)
 