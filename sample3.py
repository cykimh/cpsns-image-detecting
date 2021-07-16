import cv2
import numpy as np
from matplotlib import pyplot as plt

large_img_path = './images/youtube_screen_ios_1.png'
small_img_path = 'images/like_btn_ios_1.png'

# 검색할 큰 이미지 읽기
large_img = cv2.imread(large_img_path, 0)
large_img_width, large_img_height = large_img.shape[::-1]

new_width = int(720)
new_height = int(new_width * large_img_height / large_img_width)

print("## 큰이미지 높이,너비 리사이즈", large_img_width, large_img_height, " => ", new_width, new_height)

large_img = cv2.resize(large_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
# 읽어와서 복사??
large_img_copy = large_img.copy()

# 큰이미지 내에 찾을 템플릿이미지 읽기
small_img = cv2.imread(small_img_path, 0)
width, height = small_img.shape[::-1]
print("## 템플릿이미지 높이,너비 리사이즈", width, height)
# All the 6 methods for comparison in a list
# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
methods = ['cv2.TM_CCOEFF_NORMED']
# TM_CCOEFF_NORMED 이거를 많이 쓰는듯한데...
for meth in methods:
    img = large_img_copy.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img, small_img, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    print("상관 관계의 최소 최대 :: ", meth, min_val, max_val);
    # 사각형을 그리기위한 오른쪽 아래 지점 잡기
    bottom_right = (top_left[0] + width, top_left[1] + height)

    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    plt.subplot(121)
    plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])

    plt.subplot(122)
    plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    # 이미지 시각화
    plt.show()
