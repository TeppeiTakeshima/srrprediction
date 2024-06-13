# 必要なライブラリをインポート
import streamlit as st
import numpy as np
import pandas as pd
#機械学習ライブラリ
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

st.title("Prediction Model for Micro-TESE")
st.write("##### distributed by Teppei Takeshima @ YCURepro")
st.write("## Input Value")

#ファイルの読み込み
path = "tese_data3.csv"
df=pd.read_csv(path)


#不要な列を削除
df = df.drop(["number", "pt_id","smoking", "varicocele_3","varicocele_2","varicocele_1", "varicocele_0","varicocele","clin_varico", "BMI", "other_gene", "azf_gr", "testis_right", "testis_left"],axis = 1)

df["KS"]=df["KS"].fillna(0).astype(int)


#データを目標値と説明変数に分割
t = df["sperm"]
x = df.drop("sperm", axis= 1)


#train_test_split
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.30, random_state=0)

#XGBoostのインポート
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=3000, max_depth = 10)

model.fit(x_train, t_train)
pred = model.predict(x_test)

# 予測精度を確認
print(classification_report(t_test, pred))

# 学習済みモデルを保存
joblib.dump(model, 'srrpredict.pkl', compress=True)

# サイドバー（入力画面）
st.sidebar.header('Input Features')

age_value  = st.sidebar.slider('age (years)', min_value=0, max_value=100, step=1)
height_value = st.sidebar.slider('height (cm)', min_value=150.0, max_value=200.0, step=0.1)
weight_value = st.sidebar.slider('weight (kg)', min_value=30.0, max_value=150.0, step=0.1)
t_value = st.sidebar.slider('testosterone (ng/ml)', min_value=0.0, max_value=15.0, step=0.1)
lh_value = st.sidebar.slider('LH (mIU/ml)', min_value=0.0, max_value=100.0, step=0.1)
fsh_value =st.sidebar.slider('FSH (mIU/ml)', min_value=0.0, max_value=100.0, step=0.1)
e2_value = st.sidebar.slider('E2 (pg/ml)', min_value=0.0, max_value=100.0, step=0.1)
mean_vol = st.sidebar.slider('mean testicular volume (ml)', min_value=0.0, max_value=30.0, step=0.1)
KS = st.selectbox("KS",range(0,2))
AZF_C = st.selectbox("AZF_C",range(0,2))
Drug = st.selectbox("Drug",range(0,2))
Undescended_testis = st.selectbox("Undescended_testis",range(0,2))
Cryptozoospermia = st.selectbox("Cryptozoospermia",range(0,2))


value_df = pd.DataFrame({'data':'data', 'age':age_value, 'Height': height_value, 'Weight': weight_value, "T": t_value, "LH": lh_value, "FSH":fsh_value, "E2": e2_value, "KS": KS, "azf_c": AZF_C,  "mean_vol": mean_vol,"drug": Drug, "undescended": Undescended_testis, "cryptozoo":Cryptozoospermia }, index=[0])
value_df.set_index('data', inplace=True)

st.write(value_df)

# 予測値のデータフレーム
pred_probs = model.predict_proba(value_df)
pred_df = pd.DataFrame(pred_probs,columns=['failure','success',],index=['probability'])

st.write('## Prediction')
st.write(pred_df)

# 予測結果の出力
name = pred_df.idxmax(axis=1).tolist()
st.write('## Result')
st.write('Probability of sperm retrieval is',str(name[0]))