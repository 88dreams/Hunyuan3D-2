# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from PIL import Image
from rembg import remove, new_session


class BackgroundRemover():
    def __init__(self):
        # Use CPU only for reliability and speed - avoid GPU provider issues
        try:
            self.session = new_session(providers=['CPUExecutionProvider'])
            print("Background removal using: CPUExecutionProvider")
        except Exception as e:
            print(f"Failed to initialize background removal: {e}")
            self.session = None

    def __call__(self, image: Image.Image):
        if self.session is None:
            print("Background removal not available, returning original image")
            return image

        output = remove(image, session=self.session, bgcolor=[255, 255, 255, 0])
        return output
