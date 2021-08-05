
# 구독형 상품의 이미지 검증을 위한 이미지 프로세싱 처리

Nstation 유저들의 구독형 상품의 참여 검증을 위한 이미지 프로세싱 작업이 필요
이미지 프로세싱 작업에 필요한 Google Vision API 및 OpenCV 라이브러리 모듈 사용

## 상품 유형
1) 유튜브 영상 좋아요+구독
2) 인스타 페이지 좋아요
3) 페이스북 페이지 좋아요
4) 네이버 언론사 구독
5) 카카오톡 플친 추가


## google-vision-api (OCR)
광학 문자 인식(Optical character recognition; OCR) 기술로 텍스트를 추출함

### API 사용 관련 대쉬보드
[https://console.cloud.google.com/apis/dashboard?hl=ko&project=reflected-jet-176504](https://console.cloud.google.com/apis/dashboard?hl=ko&project=reflected-jet-176504)

### Google Vision API 가격표 및 측정

[가격 책정 | Cloud Vision API | Google Cloud](https://cloud.google.com/vision/pricing?hl=ko)

[Google Cloud Platform Pricing Calculator](https://cloud.google.com/products/calculator#id=2007e1b7-d970-4510-ad73-70b0d363b940)

## opencv-template-matching

OpenCV 라이브러리의 Template Matching 기능을 사용하여, 기준이 되는 이미지에서 원하는 template 이미지를 찾는 방식을 사용함. 
정확도 및 인식률 확인 테스트


## 서버 모듈 설치

* Python 설치(python >= 3.6)
```
$ sudo yum install -y https://repo.ius.io/ius-release-el7.rpm
$ sudo yum install -y python36u python36u-pip python36u-libs python36u-devel 
$ sudo pip3 install --upgrade pip
```
* Google Vision 모듈설치
``` 
$ sudo pip3 install google-cloud-vision==2.4.0 
```
* OpenCV Python 모듈설치
```
$ sudo pip3 install opencv-python==4.5.3.56 
```


### 참조사이트
[Google-Vision-API](https://googleapis.dev/python/vision/latest/index.html)

[OpenCV-Python](https://pypi.org/project/opencv-python/)


$ python main.py -t youtube -i dataset/youtube/G6/Screenshot_20210716-100941.png
$ python main.py -t youtube -i dataset/youtube/S4_LTE-A/Screenshot_2021-07-16-11-47-50.png

$ python main.py -t shoppinglive -i dataset/shoppinglive/shoppinglive_sample1.jpg
$ python main.py -t shoppinglive -i dataset/shoppinglive/shoppinglive_sample2.png
$ python main.py -t shoppinglive -i dataset/shoppinglive/shoppinglive_sample3.png

