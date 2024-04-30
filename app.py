import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.ecl import ECL_model, balanced_proxies
import altair as alt
import pandas as pd

# 加载模型，使用缓存，避免每次都要重复加载
@st.cache_data
def load_model():
    model = ECL_model(num_classes=7, feat_dim=128)
    proxy_num_list = get_proxies_num([84, 195, 69, 4023, 308, 659, 667])
    model_proxy = balanced_proxies(dim=128, proxy_num=sum(proxy_num_list))
    model.load_state_dict(torch.load("bestacc_model_106.pth", map_location=torch.device('cpu')), strict=True)
    model.eval()
    return model

# 计算代理数
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

# 预测图像的分类结果，为softmax值
def predict(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) # 图像预处理（改变大小，转为tensor，归一化）
    img = transform(image).unsqueeze(0) # 变为4个通道（batch,channel,width,height）
    output = model(img)
    softmax_output = F.softmax(output, dim=1)
    return softmax_output

model = load_model()

st.title('SkinWatch 皮肤癌智能辅助诊断工具')
st.warning("识别结果仅供参考，以专业人士意见为准")

uploaded_file = st.file_uploader("上传皮肤图片(.jpg)", type="jpg")

# 上传图片成功后，显示图片，进行预测
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

    # 使用Altair创建一个图表对象，指定数据源
    chart = alt.Chart(data).mark_bar().encode(
        # 设定条形图的x轴为模型预测的概率
        x='Predicted Probability',
        # 设定条形图的y轴为类别名称
        y='Class',
        # 设定条形图的颜色，如果条目类别与预测类别相符，则为橙色，否则为钢蓝色
        color=alt.condition(
            alt.datum.Class == ans,  # 判断条件：数据的类别是否为预测类别
            alt.value('orange'),  # 条件为真时的颜色
            alt.value('steelblue')  # 条件为假时的颜色
        )
    ).properties(
        width=500,  # 图表宽度
        height=300  # 图表高度
    )
    
    # 使用Streamlit显示图表
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
