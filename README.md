# choosa

* choosa는 한국 전통 회화를 학습한 생성인공지능으로 프롬프트나 이미지를 입력하면 한국 전통 화법을 적용한 그림을 생성합니다.

# 목표

* choosa는 현대 미술이 발전함 속에서 한국 전통 미술의 새로운 발견을 시도를 위해 개발되었습니다.   
* choosa는 대규모 사전학습이 된 [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)기반의 모델에 한국 전통회화의 화풍을 추가로 학습하여 전통회화 뿐아니라 두 가지의 화법이 섞인
새로운 양식을 재현 할 수 있습니다.   
* choosa는 사장되어가는 한국 전통 미술 시장 활성화와 대중들의 관심과 인식 개선을 목표로 하고 있습니다.   



# Usage

사용자는 [choosa](https://huggingface.co/spaces/calihyper/choosa)에서 **Text-to-image** 또는 **Controlnet-Canny**를 이용하여 이미지를 생성할 수 있습니다.

![image](https://user-images.githubusercontent.com/119021313/236125077-d31ef1ca-cb1d-48fb-96fa-5b7a458111b0.png)

### 1. Use ControlNet : [ControlNet-Canny](https://github.com/lllyasviel/ControlNet)의 사용여부 입니다. 사용할 경우 canny image에 이미지를 업로드 한 후 threshold를 조절해주세요.   threshold가 낮을 수록 윤곽선을 자세하게 검출합니다.   
### 2. Image : 모델이 생성한 이미지가 출력됩니다.   
### 3. prompt : 생성하고 싶은 이미지에 반영하고 싶은 내용을 입력하세요.   
* 현재 적용가능한 스타일 목록 (양옆에 < , >를 붙여주세요.)   
"trad-kor-landscape-black", "trad-kor-landscape-ink-wash-painting", "trad-kor-landscape-thick-brush-strokes", "trad-kor-plants-black", "trad-kor-plants-color"
 
### 4. negative prompt : 이미지에 제외하고 싶은 부분, 반영하고 싶지 않은 내용을 입력하세요.   
### 5. inference steps : 추론의 단계를 조절합니다. 일반적으로 높은수록 품질은 올라가지만 생성시간도 길어집니다.    
### 6. guidance scale : 1보다 클수록 프롬프트에 얼마나 밀접하게 연결되는지 영향을 줍니다. 높은 guidance scale은 저품질의 이미지를 생성합니다.    

# examples
### Text-to-Image
 * prompt : a painting of Eiffel tower in Paris "trad-kor-landscape-ink-wash-painting"   
 * inference steps: 30   
 * guidance scale : 2.7   
<img src="./img_examples/landscape-ink-wash-painting-tti.png" width="300" height="300">   

### Controlnet-Canny   
 * prompt : a painting of mountain "trad-kor-thick-brush-strokes"   
 * inference steps: 50     
 * guidance scale : 4.7    
<img src="./img_examples/thick-brush-strokes-controlnet.png" width="300" height="300">

    
    
# 일정
|Task|목표기간|세부내용|
|:---:|---|---|
|Model의 이해와 구현|2023.03.24 ~ 2023.03.29|Huggingface의 사용법에 대한 기초|
|모델 구현|2023.03.30 ~ 2023.04.07|-|
|데이터 셋 수집 및 정제|2023.03.30~ 2023.04.26|-|
|모델 완성 <br> (서비스 배포)|2023.04.24~2023.04.28|최종 서빙 모델 추출|
|문서 작성 <br> (서비스 배포)|2023.05.01~2023.05.04|-|

# 역할 분배
|주요 담당 업무|역할 상세|인원|
|:---:|:---:|:---:|
|모델 구현 및 테스트|pre-trained model을 다양한 방식으로 fine-tuning|3|
|데이터 수집 및 정제|오픈 데이터를 수집하고 데이터셋으로 활용 가능하게 정제|3|


# Reference
1. https://arxiv.org/abs/2302.05543
2. https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines
3. https://huggingface.co/blog/controlnet
4. https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img
5. https://huggingface.co/docs/diffusers/training/text_inversion
6. https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion
