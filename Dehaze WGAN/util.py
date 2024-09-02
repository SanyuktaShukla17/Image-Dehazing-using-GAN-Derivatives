## Dehazing images -------------------------

# import os

# for i in range(14, 1449):
#     file_name = '{:04d}'.format(i)
#     os.system('python haze_removal.py /run/media/aoyon/HDD_main/Code/projects/btp/Dehaze-GAN/A/{file}.png {file}'.format(file = file_name))
#     print(file_name)

# print('------------DONE------------')


## validation ----------------------------

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np 
import cv2

#number of images in the dataset
NUM_IMAGES = 14
total_ssim = 0
total_psnr = 0
psnr_weight = 1/20
ssim_weight = 1
val_image_count = NUM_IMAGES

# append all values to a CSV file

metrics_csv = open("metrics.csv", "a")
metrics_csv.write("FileNum, PSNR, SSIM \n")

for i in range(val_image_count):

    file_num = '{:04d}'.format(i)
    real_img_path = '/run/media/aoyon/HDD_main/Code/projects/btp/Dehaze-GAN/B/{file}.png'.format(file = file_num)
    generated_img_path = '/run/media/aoyon/HDD_main/Code/projects/btp/Dehaze-GAN/C/{file}.jpg'.format(file = file_num)

    real_img = cv2.imread(real_img_path, cv2.IMREAD_UNCHANGED)
    generated_img = cv2.imread(generated_img_path, cv2.IMREAD_UNCHANGED)

    psnr = peak_signal_noise_ratio(real_img, generated_img)
    ssim = structural_similarity(real_img, generated_img, channel_axis = 2)

    total_psnr = total_psnr + psnr
    total_ssim = total_ssim + ssim

    metrics_csv.write('{file_num}, {psnr}, {ssim} \n'.format(file_num=file_num, psnr=psnr, ssim=ssim))

average_psnr = total_psnr / val_image_count
average_ssim = total_ssim / val_image_count

# add to CSV
metrics_csv.write("Average, {PSNR}, {SSIM} \n".format(PSNR = average_psnr, SSIM = average_ssim))
metrics_csv.close()

print('average_psnr = {psnr}, average_ssim = {ssim}.'.format(psnr=average_psnr, ssim=average_ssim))
score = average_psnr * psnr_weight + average_ssim * ssim_weight
print(score)
