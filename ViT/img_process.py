import os
import shutil
import random
from tqdm import tqdm
from PIL import Image, ImageEnhance


def random_brightness(image):  # 亮度
    brightness = random.uniform(0.5, 1.5)
    enh_bri = ImageEnhance.Brightness(image)
    img_brightened = enh_bri.enhance(brightness)
    return img_brightened


def random_contrast(image):  # 对比度
    contrast = random.uniform(0.5, 1.5)
    enh_con = ImageEnhance.Contrast(image)
    img_contrasted = enh_con.enhance(contrast)
    return img_contrasted


def random_color(image):  # 色度
    color = random.uniform(0.5, 1.5)
    enh_col = ImageEnhance.Contrast(image)
    img_colored = enh_col.enhance(color)
    return img_colored


def random_sharpness(image):  # 锐度
    sharpness = random.uniform(0.5, 1.5)
    enh_sha = ImageEnhance.Contrast(image)
    img_sharped = enh_sha.enhance(sharpness)
    return img_sharped


def random_enhance(content_imgs):  # 扩充数据集，随机增强亮度、对比度、色度、锐度
    content_imgs_exp = 'data/train_exp2/'
    if not os.path.exists(content_imgs_exp):
        os.makedirs(content_imgs_exp)
    for cls in os.listdir(content_imgs):
        if cls == 'test':
            print(cls)
            continue
        floder_exp = content_imgs_exp + cls + '/'
        if os.path.exists(floder_exp):
            shutil.rmtree(floder_exp)
        os.makedirs(floder_exp)
        for path_img in tqdm(os.listdir(content_imgs + cls)):
            img = Image.open(content_imgs + cls + '/' + path_img)
            shutil.copy(content_imgs + cls + '/' + path_img, floder_exp + path_img)
            random_brightness(img).save(floder_exp + path_img[:-4] + '_bri.png')
            random_contrast(img).save(floder_exp + path_img[:-4] + '_con.png')
            random_color(img).save(floder_exp + path_img[:-4] + '_col.png')
            random_sharpness(img).save(floder_exp + path_img[:-4] + '_sha.png')



if __name__ == '__main__':
    content_imgs = 'data/train_test/'
    random_enhance(content_imgs)