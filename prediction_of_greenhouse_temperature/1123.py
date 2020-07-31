import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'J:\test.png', 0)
# 第二步：进行数据类型转换
img_float = np.float32(img)
# 第三步：使用cv2.dft进行傅里叶变化
dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
# 第四步：使用np.fft.fftshift将低频转移到图像中心
dft_center = np.fft.fftshift(dft)
# 第五步：定义掩模：生成的掩模中间为0周围为1
crow, ccol = int(img.shape[0] / 2), int(img.shape[1] / 2)  # 求得图像的中心点位置

# ccol

a = [0, 1, 3, 5, 10, 20, 30, ccol]
plt.figure(figsize=[10 * len(a), 10])
plt.subplot(f'1{len(a)}1')
plt.imshow(img, cmap='gray')
for i in range(1, len(a)):
    print(i)
    print(a[i - 1], a[i])

    mask = np.zeros((img.shape[0], img.shape[1], 2), np.uint8)
    mask[crow - a[i]:crow + a[i], ccol - a[i]:ccol + a[i]] = 1
    mask[crow - a[i - 1]:crow + a[i - 1], ccol - a[i - 1]:ccol + a[i - 1]] = 0
    # 第六步：将掩模与傅里叶变化后图像相乘，保留中间部分
    mask_img = dft_center * mask

    # 第七步：使用np.fft.ifftshift(将低频移动到原来的位置
    img_idf = np.fft.ifftshift(mask_img)

    # 第八步：使用cv2.idft进行傅里叶的反变化
    img_idf = cv2.idft(img_idf)

    # 第九步：使用cv2.magnitude转化为空间域内
    img_idf = cv2.magnitude(img_idf[:, :, 0], img_idf[:, :, 1])
    plt.subplot(f'1{len(a)}{i + 1}')
    plt.imshow(img_idf, cmap='gray')
# 第十步：进行绘图操作
plt.show()
