# Import packages
import streamlit as st
import altair as alt
import pandas as pd 
import numpy as np
import joblib
from transformers import pipeline

# Load pipeline
pipe_lr = joblib.load(open("emotion_detector_pipe_lr.pkl","rb"))
emoroberta_emotion_classifier = pipeline(
    "text-classification", model="arpanghoshal/EmoRoBERTa", return_all_scores=True
)

# Emojis
emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "joy":"🤗", "neutral":"😐", "sadness":"😔", "shame":"😳", "surprise":"😮"}

# Functions
def predict_emotions(text):
    result = pipe_lr.predict([text])
    return result[0]

def get_prediction_proba(text):
    result = pipe_lr.predict_proba([text])
    return result

def predict_emotions_and_score_emoroberta(text):
    emotionlabel = emoroberta_emotion_classifier(text)
    emotion_dataframe = pd.DataFrame.from_records(emotionlabel[0]).sort_values(by=["score"], ascending=False)
    emotion = emotion_dataframe.iloc[0, 0]
    score = emotion_dataframe.iloc[0, 1]
    return emotion_dataframe, emotion, score

def main():
    st.title("Emotion Detection App")
    with st.form(key='emotion_detection_form'):           
        raw_text = st.text_area("Type here")
        submit_text = st.form_submit_button(label = 'Submit')
    if submit_text:
        st.subheader("Predictions from LR model trained on labeled data (8 emotions)")
        col1,col2 = st.columns(2)
        prediction = predict_emotions(raw_text)
        prediction_probability = get_prediction_proba(raw_text)
        with col1:
            st.success("Original Text")
            st.write(raw_text)
            st.success("Emotion")    
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{} {}".format(prediction,emoji_icon)) 
            st.success("Emotion Score") 
            st.write("{:.4f}".format(np.max(prediction_probability)))

        with col2:
            st.success("Prediction Probability") 
            proba_df = pd.DataFrame(prediction_probability,columns = pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions","probability"]
            fig = alt.Chart(proba_df_clean).mark_bar().encode(y=alt.Y('emotions', sort='-x'),x='probability',color='emotions')
            st.altair_chart(fig,use_container_width=True) 
            
        st.markdown("***")
        
        st.subheader("Predictions from EmoRoBERTa - BERT based pre-trained model (28 emotions)")
        
        col3,col4 = st.columns(2)
        emotion_dataframe_emoroberta, predicted_emotion_emoroberta, prediction_probability_emoroberta = predict_emotions_and_score_emoroberta(raw_text)
        with col3:
            st.success("Emotion")    
            st.write(predicted_emotion_emoroberta) 

        with col4:
            st.success("Emotion Score") 
            st.write("{:.4f}".format(np.max(prediction_probability_emoroberta)))
        st.success("Prediction Probability")
        emotion_dataframe_emoroberta.columns = ["emotions","probability"]  
        fig = alt.Chart(emotion_dataframe_emoroberta).mark_bar().encode(y=alt.Y('emotions', sort='-x'),x='probability',color='emotions')
        st.altair_chart(fig,use_container_width=True)

if __name__ == '__main__':
    main()