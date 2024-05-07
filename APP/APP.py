import streamlit as st
from predict_DL_2 import predict as predict_DL
from predict_ML_2 import predict as predict_ML
import pandas as pd

def main():
    # Define the title
    title_html = "<h1><span style='color:#F50000; font-size:59px; font-family:Arial, sans-serif'>Moroc</span><span style='color:#089301; font-size:56px; font-family:Arial, sans-serif'>Can</span><span style='color:#F9F6F6; font-size:49px; font-family:Arial, sans-serif'> Musique</span></h1>"

    # Use columns layout to display the image and title in the same row
    col1, col2 = st.columns([1, 3])
    with col1:
      img_url="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/Flag_of_Morocco.svg/1280px-Flag_of_Morocco.svg.png"
      st.image(img_url)
    with col2:
      st.markdown(title_html, unsafe_allow_html=True)

    page_img = '''
          <style>
          [data-testid="stAppViewContainer"]{
            background-image: url("https://oubaditravel.com/wp-content/uploads/2020/01/5-days-Marrakech-to-Merzouga-1024x575.jpg");
            background-size: cover;
            }
          [data-testid="stHeader"]{
            background-color: rgba(0,0,0,0);
            }
          </style>
        '''
    st.markdown(page_img, unsafe_allow_html=True)

  
    st.markdown("<h3 style='font-weight: bold; color: white;'>Choose Prediction Method:</h3>", unsafe_allow_html=True)
    model_choice = st.selectbox("", ("Deep Learning (DL)", "Machine Learning (ML)"), format_func=lambda x: f"{x}")

    if model_choice == "Deep Learning (DL)":
        predict_function = predict_DL
        # Define the appropriate on_file_upload function for DL
        def on_file_upload():
            predicted_genre, confidence = predict_function(uploaded_file)
            st.write("### Predicted Genre:")
            st.write(f"#### Genre: {predicted_genre}")
            st.write(f"#### Confidence: {confidence:.4f}%")
            # Change background image based on predicted genre for DL
            if predicted_genre == 'CHAABI':
                page_img = '''
                            <style>
                            [data-testid="stAppViewContainer"]{
                              background-image: url("https://www.moroccopedia.com/wp-content/uploads/2017/03/ahwach-680x420.jpg");
                              background-size: cover;
                              }
                            [data-testid="stHeader"]{
                              background-color: rgba(0,0,0,0);
                               }
                            </style>
                          '''
            elif predicted_genre == 'CHARKI':
                page_img = '''
                            <style>
                            [data-testid="stAppViewContainer"]{
                              background-image: url("https://www.musicinafrica.net/sites/default/files/images/article/202401/image-asset.jpeg");
                              background-size: cover;
                              }
                            [data-testid="stHeader"]{
                              background-color: rgba(0,0,0,0);
                               }
                            </style>
                          '''
            elif predicted_genre == 'RAP':
                page_img = '''
                            <style>
                            [data-testid="stAppViewContainer"]{
                              background-image: url("https://scenenoise.com/Content/Articles/Big_image/1b3017a3-5995-4df4-ab1c-218d2515a891.jpg");
                              background-size: cover;
                              }
                            [data-testid="stHeader"]{
                              background-color: rgba(0,0,0,0);
                               }
                            </style>
                          '''
            elif predicted_genre == 'RAI':
                page_img = '''
                            <style>
                            [data-testid="stAppViewContainer"]{
                              background-image: url("https://ds.static.rtbf.be/article/image/1248x702/9/f/3/b21f5ea983b94eb4be54e233aaa6a94f-1669907970.jpg");
                              background-size: cover;
                              }
                            [data-testid="stHeader"]{
                              background-color: rgba(0,0,0,0);
                               }
                            </style>
                          '''
            elif predicted_genre == 'TACHLHIT':
                page_img = '''
                            <style>
                            [data-testid="stAppViewContainer"]{
                              background-image: url("https://sudestmaroc.com/wp-content/uploads/2022/06/Ahwach-costume-1.jpg");
                              background-size: cover;
                              }
                            [data-testid="stHeader"]{
                              background-color: rgba(0,0,0,0);
                               }
                            </style>
                          '''
            elif predicted_genre == 'TAKTOKA':
                page_img = '''
                            <style>
                            [data-testid="stAppViewContainer"]{
                              background-image: url("https://cdn.communesmaroc.com/media/Ksar-El-Kebir/news/2015/08/1440402702.jpg");
                              background-size: cover;
                              }
                            [data-testid="stHeader"]{
                              background-color: rgba(0,0,0,0);
                               }
                            </style>
                          '''
            elif predicted_genre == 'GNAWA':
                page_img = '''
                            <style>
                            [data-testid="stAppViewContainer"]{
                              background-image: url("https://assets-global.website-files.com/5d6d8f89b695c7eeede0e3a3/5dcc9e085473766e7ee140ba_morocco-sahara-desert-gnawa-music.jpeg");
                              background-size: cover;
                              }
                            [data-testid="stHeader"]{
                              background-color: rgba(0,0,0,0);
                               }
                            </style>
                          '''
            st.markdown(page_img, unsafe_allow_html=True)
            
    elif model_choice == "Machine Learning (ML)":
        predict_function = predict_ML
        # Define the appropriate on_file_upload function for ML
        def on_file_upload():
            predictions = predict_function(uploaded_file)
            st.write("### Predictions:")
            
            # Create a list of dictionaries for table data
            table_data = [{"Model": prediction['Model'], "Prediction": prediction['Prediction']} for prediction in predictions]
            
            # Create a DataFrame from table_data
            df = pd.DataFrame(table_data)

            # Apply styling to the DataFrame
            st.dataframe(df.style.set_properties(**{'text-align': 'center', 'background-color': '#f2f2f2', 'color': 'black'})
                                  .applymap(lambda x: f"background-color: {'#dddddd' if x != '' else ''}", subset=['Model', 'Prediction']))

            # Maximize predictions
            prediction_counts = df['Prediction'].value_counts()
            max_prediction = prediction_counts.idxmax()
            st.write(f"#### Genre: {max_prediction}")
            # Change background image based on maximized prediction for ML
            if max_prediction == 'CHAABI':
                page_img = '''
                            <style>
                            [data-testid="stAppViewContainer"]{
                              background-image: url("https://www.moroccopedia.com/wp-content/uploads/2017/03/ahwach-680x420.jpg");
                              background-size: cover;
                              }
                            [data-testid="stHeader"]{
                              background-color: rgba(0,0,0,0);
                               }
                            </style>
                          '''
            elif max_prediction == 'CHARKI':
                page_img = '''
                            <style>
                            [data-testid="stAppViewContainer"]{
                              background-image: url("https://www.musicinafrica.net/sites/default/files/images/article/202401/image-asset.jpeg");
                              background-size: cover;
                              }
                            [data-testid="stHeader"]{
                              background-color: rgba(0,0,0,0);
                               }
                            </style>
                          '''
            elif max_prediction == 'RAP':
                page_img = '''
                            <style>
                            [data-testid="stAppViewContainer"]{
                              background-image: url("https://scenenoise.com/Content/Articles/Big_image/1b3017a3-5995-4df4-ab1c-218d2515a891.jpg");
                              background-size: cover;
                              }
                            [data-testid="stHeader"]{
                              background-color: rgba(0,0,0,0);
                               }
                            </style>
                          '''
            elif max_prediction == 'RAI':
                page_img = '''
                            <style>
                            [data-testid="stAppViewContainer"]{
                              background-image: url("https://ds.static.rtbf.be/article/image/1248x702/9/f/3/b21f5ea983b94eb4be54e233aaa6a94f-1669907970.jpg");
                              background-size: cover;
                              }
                            [data-testid="stHeader"]{
                              background-color: rgba(0,0,0,0);
                               }
                            </style>
                          '''
            elif max_prediction == 'TACHLHIT':
                page_img = '''
                            <style>
                            [data-testid="stAppViewContainer"]{
                              background-image: url("https://sudestmaroc.com/wp-content/uploads/2022/06/Ahwach-costume-1.jpg");
                              background-size: cover;
                              }
                            [data-testid="stHeader"]{
                              background-color: rgba(0,0,0,0);
                               }
                            </style>
                          '''
            elif max_prediction == 'TAKTOKA':
                page_img = '''
                            <style>
                            [data-testid="stAppViewContainer"]{
                              background-image: url("https://cdn.communesmaroc.com/media/Ksar-El-Kebir/news/2015/08/1440402702.jpg");
                              background-size: cover;
                              }
                            [data-testid="stHeader"]{
                              background-color: rgba(0,0,0,0);
                               }
                            </style>
                          '''
            elif max_prediction == 'GNAWA':
                page_img = '''
                            <style>
                            [data-testid="stAppViewContainer"]{
                              background-image: url("https://assets-global.website-files.com/5d6d8f89b695c7eeede0e3a3/5dcc9e085473766e7ee140ba_morocco-sahara-desert-gnawa-music.jpeg");
                              background-size: cover;
                              }
                            [data-testid="stHeader"]{
                              background-color: rgba(0,0,0,0);
                               }
                            </style>
                          '''
            st.markdown(page_img, unsafe_allow_html=True)

    
    #st.markdown("<style>div.row-widget.stRadio > div{color: white;}</style>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-weight: bold; color: white;'>Upload music audio:</h3>", unsafe_allow_html=True)
  
    uploaded_file = st.file_uploader("", type=["wav"], accept_multiple_files=False, key="audio_uploader")
    st.set_option('deprecation.showfileUploaderEncoding', False)

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        # Automatically trigger prediction when file is uploaded
        on_file_upload()
                
if __name__ == "__main__":
    main()
