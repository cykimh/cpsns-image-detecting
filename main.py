# -*- coding: utf-8 -*-

import os
import cv2
import sys
import argparse
import io
import datetime
import numpy as np
from google.cloud import vision

# from matplotlib import pyplot as plt

# google-vision-api 관련 key file path
KEY_FILE = 'reflected-jet-176504-4b9d781b0f09.json'
IMAGE_DIR_PATH = 'images'
SCRIPT_DIR_PATH = ''


def main(arguments):
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

	# 입력받을 인자 등록
	parser.add_argument('-i', '--image_path', help="캡처 이미지 파일 경로", required=True)
	parser.add_argument('-t', '--sns_type', help="CPSNS 타입", required=True)
	parser.add_argument('-f', '--find_text', help="찾을 텍스트", nargs='+', required=False)
	parser.add_argument('-e', '--env', default='dev', help="개발환경은 뭔가", required=False)
	parser.add_argument('-p', '--script_dir_path', default='', help="스크립트 디렉토리 경로", required=False)
	parser.add_argument('-q', '--image_dir_path', default='images', help="이미지 디렉토리 경로", required=False)

	args = parser.parse_args(arguments)

	sns_type = args.sns_type
	image_path = args.image_path
	find_text = args.find_text

	global IMAGE_DIR_PATH
	IMAGE_DIR_PATH = args.image_dir_path
	global SCRIPT_DIR_PATH
	SCRIPT_DIR_PATH = args.script_dir_path

	# GOOGLE Vision API KEYFILE
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = SCRIPT_DIR_PATH + '/' + KEY_FILE

	"""
	# 1. 유튜브 영상 좋아요 + 구독 (영상 좋아요 + 채널 구독))
	# 2. 인스타 페이지 좋아요 (페이지 팔로우) 
	# 3. 페이스북 페이지 좋아요 (좋아요 + 참여검증텍스트)
	# 4. 카카오 플친추가 (참여검증텍스트)
	# 5. 네이버 언론사 구독 (네이버 언론사 구독)
	# 6. 쇼핑라이브 방송 알림 설정 (참여검증텍스트)
	"""

	if sns_type == 'youtube':
		sns = Youtube(image_path, find_text)
	elif sns_type == 'insta':
		sns = Instagram(image_path, find_text)
	elif sns_type == 'facebook':
		sns = Facebook(image_path, find_text)
	elif sns_type == 'kakao':
		sns = Kakao(image_path, find_text)
	elif sns_type == 'media':
		sns = Media(image_path, find_text)
	elif sns_type == 'shoppinglive':
		sns = Shoppinglive(image_path, find_text)

	result = sns.run_check()

	# 최종값 Return
	print(result)


# 템플릿매칭 함수
def template_matching(source_img_path, template_img_path, resize_width, correlation_rate):
	# 비교를 위한 리스트에 있는 모든 6가지 방법
	# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED'
	# , 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
	# print("Start template_matching", source_img_path, template_img_path, resize_width, correlation_rate)
	methods = ['cv2.TM_CCOEFF_NORMED']

	# 첨부파일 이미지 읽기 - flag(-1: IMREAD_COLOR 0: IMREAD_GRAYSCALE 1: IMREAD_UNCHANGED)
	source_img = cv2.imread(source_img_path, cv2.IMREAD_GRAYSCALE)

	if source_img is None:
		return {'result': "error", 'result_msg': 'Not Found Source Image'}

	# (크기, 정밀도, 채널) 값을 반환
	s_img_width, s_img_height = source_img.shape[::-1]

	# 첨부 이미지파일 검색을 위한 리사이즈
	new_height = int(resize_width * s_img_height / s_img_width)

	# print("## 큰이미지 높이,너비 리사이즈", large_img_width, large_img_height, " => ", resize_width, new_height)
	source_img = cv2.resize(source_img, (resize_width, new_height), interpolation=cv2.INTER_AREA)
	source_img_copy = source_img.copy()

	# 이미지 매칭 결과
	found = None
	found_flag = False

	# 찾을 이미지 List 형식
	if not isinstance(template_img_path, list):
		template_img_path = [template_img_path]

	for t_image_path in template_img_path:

		# 찾을 템플릿이미지 읽기
		template_img = cv2.imread(t_image_path, cv2.IMREAD_GRAYSCALE)
		if template_img is None:
			return {'result': "error", 'result_msg': 'Not Found Template Image'}

		t_img_width, t_img_height = template_img.shape[::-1]

		# multiple scale로 검사 (0.2~1.0)
		for scale in np.linspace(0.5, 1.5, 20):

			t_img = cv2.resize(template_img, (int(t_img_width * scale), int(t_img_height * scale)),
							   interpolation=cv2.INTER_AREA)

			# Method 별로 확인
			for meth in methods:
				img = source_img_copy.copy()
				method = eval(meth)

				# 템플릿 매칭 적용
				res = cv2.matchTemplate(img, t_img, method)

				min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
				# print("template Matching Result ( Max_Value :", max_val, " \t Scale : ", scale, ")")

				if found is None or max_val > found[0]:

					# 만약 방법이 TM_SQDIFF나 TM_SQDIFF_NORMED라면, 최소를 취한다
					if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
						top_left = min_loc
					else:
						top_left = max_loc

					# cv.minMaxLoc(res)이 상관 관계의 최소 및 최대 결과를 계산합니다.
					# min_val및 둘 다 max_val범위에 있으므로 가 1.0 [-1.0, 1.0]이면
					# max_val100% 일치로, max_val0.5이면 50% 일치로 간주하는 식입니다.

					# 매칭 사각형 지점잡고 그림
					bottom_right = (top_left[0] + t_img_width, top_left[1] + t_img_height)
					result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
					cv2.rectangle(result_img, top_left, bottom_right, (0, 0, 255), 5)

					found = (max_val, max_loc, scale, result_img)
					if max_val > correlation_rate:
						found_flag = True
						break

			if found_flag:
				break

		if found_flag:
			break
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

	# print(found)
	# 찾은 결과 이미지 저장
	result_image_path = ''
	if found is not None:
		result_image_path = IMAGE_DIR_PATH + '/results/found_image_' + datetime.datetime.now().strftime(
			"%Y%m%d%H%M%S") + '.png'
		cv2.imwrite(result_image_path, found[3])

	# 제일 매칭 상관계수가 높은 것을 반환
	return {'result': "success", 'correlation_rate': found[0], 'found_image': result_image_path}


# Vision API OCR 함수
def vision_api_ocr(img_path):
	"""Detects text in the file."""
	client = vision.ImageAnnotatorClient()

	with io.open(img_path, 'rb') as img_file:
		content = img_file.read()

	if content is None:
		return {'result': "error", 'result_msg': 'Cant Read Upload Image'}

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

	# OCR을 통해서 찾은 텍스트 리스트 반환
	return {'result': 'success', 'detect_texts': de_text}


class Sns:
	"""Super Class"""

	def __init__(self, img_path, text_list):
		# 첨부된 이미지파일 경로
		self.img_path = img_path

		# OCR Option
		self.text_list = text_list

		# OpenCV Option
		self.resize_image_width = 720
		self.similarity_rate = 0.94

		self.template_img_path = []
		return

	def detect_text(self):  # Vision API OCR 텍스트 문자 확인

		ocr_result = vision_api_ocr(self.img_path)

		is_detect = True
		for input_text in self.text_list:
			if not any(input_text in s for s in ocr_result['detect_texts']):
				is_detect = False

		# {find}
		text_result = {}
		text_result['result'] = ocr_result['result']
		text_result['text_result_yn'] = 'y' if is_detect else 'n'
		text_result['found_text'] = ','.join(str(s) for s in ocr_result['detect_texts'])

		return text_result

	def detect_image(self):

		is_detect = False
		image_result = {}

		matching_result = template_matching(self.img_path, self.template_img_path, self.resize_image_width,
											self.similarity_rate)

		if matching_result['result'] == 'success':
			is_detect = True

		if matching_result['result'] == "success":
			if bool(matching_result['correlation_rate']) and float(
					matching_result['correlation_rate']) > self.similarity_rate:
				# 이미지검사결과(image_result_yn)
				# 이미지_매칭_결과이미지(find_result_image)
				# 이미지 일치 상관계수(correlation_rate)
				image_result['image_result_yn'] = 'y' if is_detect else 'n'
				image_result['find_image'] = ','.join(str(s) for s in self.template_img_path)
				image_result['found_image'] = matching_result['found_image']
				image_result['correlation_rate'] = round(float(matching_result['correlation_rate']), 4)
				image_result['result_msg'] = matching_result['result']
		else:
			image_result['result'] = matching_result['result']
			image_result['result_msg'] = matching_result['result_msg']
			image_result['image_result_yn'] = 'n'

		return image_result

	def run_check(self):

		return


# 1. 유튜브 영상 좋아요 + 구독
class Youtube(Sns):

	def __init__(self, img_path, text_list):
		super().__init__(img_path, text_list)

		self.template_img_path = [IMAGE_DIR_PATH + '/assets/youtube/like_btn_light.png',
								  IMAGE_DIR_PATH + '/assets/youtube/like_btn_dark.png']  # 좋아요 이미지 경로
		return

	def run_check(self):
		result = {}

		# text_result = self.detect_text()  # Vision API OCR 텍스트 문자 확인
		image_result = self.detect_image()  # OpenCV Template Matching 이미지 유무 확인

		# result.update(text_result)
		result.update(image_result)

		return result


# 2. 인스타 페이지 좋아요 (페이지 팔로우)
class Instagram(Sns):

	def __init__(self, img_path, text_list):
		super().__init__(img_path, text_list)

		self.template_img_path = [IMAGE_DIR_PATH + '/assets/instagram/follow_btn_light.png',
								  IMAGE_DIR_PATH + '/assets/instagram/follow_btn_dark.png']  # 팔로우 이미지 경로
		return

	def run_check(self):
		result = {}

		# text_result = self.detect_text()  # Vision API OCR 텍스트 문자 확인
		image_result = self.detect_image()  # OpenCV Template Matching 이미지 유무 확인

		# result.update(text_result)
		result.update(image_result)

		return result


# 3. 페이스북 페이지 좋아요 (좋아요 + 참여검증텍스트)
class Facebook(Sns):

	def __init__(self, img_path, text_list):
		# 첨부된 이미지파일 경로
		super().__init__(img_path, text_list)

		self.template_img_path = [IMAGE_DIR_PATH + '/assets/facebook/like_btn_light.png']
		# IMAGE_DIR_PATH + '/assets/facebook/like_btn_dark.png']  # 좋아요 이미지 경로
		return

	def run_check(self):
		result = {}

		text_result = self.detect_text()  # Vision API OCR 텍스트 문자 확인
		image_result = self.detect_image()  # OpenCV Template Matching 이미지 유무 확인

		result.update(text_result)
		result.update(image_result)

		result['result_yn'] = 'y' if text_result['text_result_yn'] is 'y' and image_result[
			'image_result_yn'] is 'y' else 'n'

		return result


# 4. 카카오 플친추가 (참여검증텍스트)
class Kakao(Sns):
	def __init__(self, img_path, text_list):
		super().__init__(img_path, text_list)

		return

	def run_check(self):
		return


# 5. 네이버 언론사 구독 (네이버 언론사 구독)
class Media(Sns):
	def __init__(self, img_path, text_list):
		super().__init__(img_path, text_list)

		return

	def run_check(self):
		return


# 6. 쇼핑라이브 방송 알림 설정 (참여검증텍스트)
class Shoppinglive(Sns):
	def __init__(self, img_path, text_list):
		super().__init__(img_path, text_list)

		self.template_img_path = [IMAGE_DIR_PATH + '/assets/shoppinglive/alarm_btn_light.png',
								  IMAGE_DIR_PATH + '/assets/shoppinglive/alarm_btn_dark.png']
		# IMAGE_DIR_PATH + '/assets/facebook/like_btn_dark.png']  # 좋아요 이미지 경로
		return

	def run_check(self):
		result = {}

		# text_result = self.detect_text()  # Vision API OCR 텍스트 문자 확인
		image_result = self.detect_image()  # OpenCV Template Matching 이미지 유무 확인

		# result.update(text_result)
		result.update(image_result)

		return result


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
