import streamlit as st
import pandas as pd
import numpy as np
import cv2
import PIL.Image as Image
#model
import geffnet
import torch.nn as nn
import torch
from torchvision import transforms

#https://zzsza.github.io/mlops/2021/02/07/python-streamlit-dashboard/
#https://github.com/SoheeJeong/streamlit-example/blob/master/streamlit_app.py

st.title('퍼스널컬러 기반 패션 인플루언서 추천')

@st.cache
def load_model():
    model = geffnet.create_model('efficientnet_b2', pretrained=True)
    model.classifier = nn.Sequential(
            nn.Linear(1408,512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128,8))
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    return model

def detect_face(image):
    image = cv2.resize(image,dsize=None,fx=1.0,fy=1.0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cascade 얼굴 탐지 알고리즘 
    cascade_filename = 'haarcascade_frontalface_alt.xml' # 가중치 파일 경로
    cascade = cv2.CascadeClassifier(cascade_filename) # 모델 불러오기
    results = cascade.detectMultiScale(gray,            # 입력 이미지
                                scaleFactor= 1.5,# 이미지 피라미드 스케일 factor
                                minNeighbors=5,  # 인접 객체 최소 거리 픽셀
                                minSize=(20,20)  # 탐지 객체 최소 크기
                                )     
    # 결과값 = 탐지된 객체의 경계상자 list
    if len(results)>0:                                                                        
        for box in results:
            x, y, w, h = box # 좌표 추출
            cv2.rectangle(image, (x,y), (x+w, y+h), (255,255,255), thickness=2)
        # 얼굴 부분만 크롭된 이미지 리턴    
        cropped_img = image[y:y+h,x:x+w]
        return cropped_img
    else:
        return image

def predict(model,image):
    data = cv2.resize(image, (224,224),interpolation = cv2.INTER_AREA)
    data = torch.from_numpy(data).float()
    data = torch.reshape(data,(1,3,224,224))
    output = model(data) #1x8 vector
    user_vector = output.detach().numpy()
    _, user_cls = torch.max(output, 1)
    # print(user_vector.shape, idx_to_color[user_cls.item()]) 
    return user_vector, user_cls.item()

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def main():
    # 사용자 이미지 로드
    # 만약 이미지를 업로드 했다면 원본 이미지를 업로드이미지로 설정, 아니라면 데모 이미지로 설정
    image_uploaded = st.file_uploader("본인의 얼굴이 나온 이미지를 업로드해주세요")
    image_origin = Image.open(image_uploaded) if image_uploaded else Image.open('demo.jpg')
    image_origin = np.array(image_origin.convert('RGB'))
    st.image(image_origin)

    # 얼굴 탐지
    face_image = detect_face(image_origin)

    # 모델 추론
    model = load_model()
    idx_to_color = { #인덱스-클래스 매핑
            0: "봄웜브라이트",
            1: "봄웜라이트",
            2: "여름쿨라이트",
            3: "여름쿨뮤트",
            4: "가을웜뮤트",
            5: "가을웜딥",
            6: "겨울쿨딥",
            7: "겨울쿨브라이트"
        }
    # 개인 퍼스널컬러
    user_vector,user_cls = predict(model, face_image)
    st.text("당신의 퍼스널컬러는 <"+idx_to_color[user_cls]+"> 입니다.")

    # 추천 알고리즘 실행
    if st.button("나만의 패션 인플루언서 추천받기"):
        # 인플루언서 정보 로드
        influencer_list = pd.read_csv('influencer_list.csv')
        result_vector_list = np.load('result_vector_list.npy') #(100,80)
        influencer_color_list = np.load('influencer_color_list.npy') #(100,)
        # 유저 벡터와 인플루언서 벡터 사이의 유사도 계산
        result_vector_list = result_vector_list.reshape((100,10,8))
        scores_list = np.matmul(user_vector,result_vector_list.transpose(1,2,0)) 
        scores_list = scores_list.reshape((10,100))
        # 각 인플루언서 별 10차원 스코어 벡터 -> 합을 구해서 최종 스코어 도출
        final_scores_list = sum(scores_list)
        # 가장 스코어가 높은 인플루언서 3명 선정. 추가 정보로 인플루언서의 퍼스널컬러를 제공.
        influencer_idxs = largest_indices(final_scores_list,3)
        influencer_names = [influencer_list['id'][idx] for idx in influencer_idxs]
        influencer_colors = [idx_to_color[influencer_color_list[idx]] for idx in influencer_idxs[0]]
        st.text("인플루언서: "+influencer_names[0]+", 인플루언서의 퍼스널컬러: "+influencer_colors)

if __name__ == "__main__":
    main()