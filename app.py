import cv2
import numpy as np
import streamlit as st
import torch
import albumentations as albu
from PIL import Image
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import pad, unpad
from people_segmentation.pre_trained_models import create_model

MAX_SIZE = 512

@st.cache(allow_output_mutation=True)
def cached_model():
    model = create_model('Unet_2020-07-20')
    model.eval()
    return model

model = cached_model()

transform = albu.Compose([
    albu.LongestMaxSize(max_size=MAX_SIZE),
    albu.Normalize()
])

st.title('세상에서 가장 쉬운 AI 누끼따기')

file = st.file_uploader('사람 사진을 선택해주세요', type=['jpg', 'jpeg', 'png'])

if file is not None:
    img_ori = np.array(Image.open(file).convert('RGB'))
    st.image(img_ori, caption='Original')

    img_input = transform(image=img_ori)['image']
    img_input, pads = pad(img_input, factor=MAX_SIZE, border=cv2.BORDER_CONSTANT)
    img_input = torch.unsqueeze(tensor_from_rgb_image(img_input), 0)

    with torch.no_grad():
        pred = model(img_input)[0][0]

    mask = (pred > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)

    h, w, _ = img_ori.shape
    mask_1 = cv2.resize(mask, (w, h))
    mask_3 = cv2.cvtColor(mask_1, cv2.COLOR_GRAY2RGB)

    dst = cv2.addWeighted(img_ori, 1, (mask_3 * (0, 255, 0)).astype(np.uint8), 0.5, 0)

    result = np.dstack((img_ori, mask_1 * 255))

    st.image(mask_1 * 255, caption='Mask')
    st.image(dst, caption='Image + mask')
    st.image(result, caption='Result')
