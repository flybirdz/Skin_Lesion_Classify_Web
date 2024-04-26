import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.ecl import ECL_model, balanced_proxies
import altair as alt
import pandas as pd


@st.cache_data
def load_model():
    model = ECL_model(num_classes=7, feat_dim=128)
    proxy_num_list = get_proxies_num([84, 195, 69, 4023, 308, 659, 667])
    model_proxy = balanced_proxies(dim=128, proxy_num=sum(proxy_num_list))
    model.load_state_dict(torch.load("bestacc_model_106.pth", map_location=torch.device('cpu')), strict=True)
    model.eval()
    return model

def get_proxies_num(cls_num_list):
    ratios = [max(np.array(cls_num_list)) / num for num in cls_num_list]
    prototype_num_list = []
    for ratio in ratios:
        if ratio == 1:
            prototype_num = 1
        else:
            prototype_num = int(ratio // 10) + 2
        prototype_num_list.append(prototype_num)
    assert len(prototype_num_list) == len(cls_num_list)
    return prototype_num_list

def predict(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(image).unsqueeze(0)
    output = model(img)
    softmax_output = F.softmax(output, dim=1)
    return softmax_output

model = load_model()

st.title('SkinWatch 皮肤癌智能辅助诊断工具')
st.warning("识别结果仅供参考，以专业人士意见为准")

uploaded_file = st.file_uploader("上传皮肤图片(.jpg)", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("AI识别中...")
    with st.spinner('AI识别中...'):
        output = predict(image)
        predicted_class = torch.argmax(output, dim=1).item()

    ans_list = ["血管损伤", "鲍恩病", "纤维瘤", "黑色素痣", "基底细胞癌", "良性角化病", "黑色素瘤"]
    ans = ans_list[predicted_class]
    
    st.write("诊断为：", ans)

    data = pd.DataFrame({'Class': ans_list, 'Predicted Probability': output[0].detach().numpy()})

    chart = alt.Chart(data).mark_bar().encode(
        x='Predicted Probability',
        y='Class',
        color=alt.condition(
            alt.datum.Class == ans,
            alt.value('orange'), 
            alt.value('steelblue')
        )
    ).properties(
        width=500,
        height=300
    )

    st.altair_chart(chart)


graph_img = Image.open("./data/net_graph.png")
train_img = Image.open("./data/train_result.png")

with open("models/ecl.py", "r", encoding="UTF-8") as file:
    code = file.read()


col1, col2 = st.columns(2)


with col1:
    st.subheader("模型信息")
    st.image(graph_img, caption='网络图')
with col2:
    st.subheader("训练信息")
    st.image(train_img, caption='训练结果')

with st.expander("模型代码"):
    st.code(code, language="python")