from PIL import Image
import os

test_img_path = '/home/idec2/brain_mri/dataset/brain_mri/Training/glioma/Tr-glTr_0009.jpg'  # 실제 있는 이미지 경로로 수정
img = Image.open(test_img_path)
img.show()  # or img.convert('RGB') if needed

