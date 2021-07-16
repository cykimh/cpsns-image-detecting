import os
import cv2
import sys
import argparse
# from matplotlib import pyplot as plt

KEY_PATH = 'reflected-jet-176504-4b9d781b0f09.json'


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--infile', help="Input file", type=argparse.FileType('r'))
    parser.add_argument('-o', '--outfile', help="Output file", default=sys.stdout, type=argparse.FileType('w'))

    args = parser.parse_args(arguments)

    print("arguments", args)

    # key path
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY_PATH

    # 1. 유튜브 영상 좋아요 + 구독
    # 2. 인스타 페이지 좋아요
    # 3. 페이스북 페이지 좋아요
    # 4. 네이버 언론사 구독
    # 5. 카카오 플친추가
    sns_type = 'youtube'

    if sns_type == 'youtube':
        print('Youtube')
        img_path = './images/youtube_screen.png'
        youtube = Youtube(img_path, ["구독중"])
        youtube.run_check()
    elif sns_type == 'instagram':
        print('instagram')
    elif sns_type == 'facebook':
        print('facebook')
    elif sns_type == 'naver_news':
        print('naver_news')
    elif sns_type == 'kakao_plus':
        print('kakao_plus')
    else:
        print("상품없음")


# 템플릿매칭 함수
def template_matching(large_img_path, small_img_path, new_width=720):
    # 검색할 큰 이미지 읽기
    large_img = cv2.imread(large_img_path, 0)
    large_img_width, large_img_height = large_img.shape[::-1]

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
    # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    # 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
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

        # cv.minMaxLoc(res)이 상관 관계의 최소 및 최대 결과를 계산합니다.
        # 단지 max_val 잘 일치하는지 알려주 는 데 사용 합니다.
        # min_val및 둘 다 max_val범위에 있으므로 가 1.0 [-1.0, 1.0]이면 max_val100% 일치로,
        # max_val0.5이면 50% 일치로 간주하는 식입니다.
        print("상관 관계의 최소 최대 :: ", meth, min_val, max_val)
        # 사각형을 그리기위한 오른쪽 아래 지점 잡기
        bottom_right = (top_left[0] + width, top_left[1] + height)

        cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 3)

        # 찾은 결과 이미지 저장
        cv2.imwrite('images/found_screen.png', img)

        """
        plt.subplot(121)
        plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])

        plt.subplot(122)
        plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        # 이미지 시각화
        plt.show()
        """

        # {correlation_rate, matching_img, }
        return max_val, img


# Vision API OCR 함수
def vision_api_ocr(img_path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(img_path, 'rb') as img_file:
        content = img_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    de_text = []
    for text in texts:
        # print('\n"{}"'.format(text.description))
        de_text.append(text.description)
        # vertices = (['({},{})'.format(vertex.x, vertex.y)
        #              for vertex in text.bounding_poly.vertices])
        #
        # print('bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return de_text


# 1. 유튜브 영상 좋아요 + 구독
class Youtube:
    def __init__(self, img_path, text_list):
        self.capture_img_path = img_path
        # OCR Option
        self.text_list = text_list

        # OpenCV Option
        self.fixed_img_width = 720
        self.similarity_rate = 0.9
        self.like_btn_img_path = 'images/like_btn_ios_1.png'  # 좋아요 이미지 경로

    def detect_text(self):  # Vision API OCR 텍스트 문자 확인
        de_text = vision_api_ocr(self.capture_img_path)

        is_find = True
        for input_text in self.text_list:
            if any(input_text in s for s in de_text):
                print("찾았다:", input_text)
            else:
                print("못찾았다:", input_text)
                is_find = False

        # TODO : 필요한 인자들만 dictionary 응답
        # {find}

        return {"result": is_find}

    def detect_image(self):

        # TODO : 다크, 라이트 테마에 맞게 찾을 이미지 변경
        kk = template_matching(self.capture_img_path, self.like_btn_img_path, self.fixed_img_width)

        # TODO : 상관관계 계수비교

        # TODO : 필요한 인자들만 dictionary 로 응답

        return {"result": kk}

    def run_check(self):

        result = {}

        text_result = self.detect_text()  # Vision API OCR 텍스트 문자 확인
        image_result = self.detect_image()  # OpenCV Template Matching 이미지 유무 확인

        result.update(text_result)
        result.update(image_result)

        return result


# 2. 인스타 페이지 좋아요
class Instagram:
    def __init__(self):
        return


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
