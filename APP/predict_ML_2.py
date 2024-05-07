import pandas as pd
import numpy as np
import librosa
import pickle
import catboost as cb
import xgboost as xgb

def predict(file_path):
            audio_data, sr = librosa.load(file_path) #, offset=0, duration=30)
            audio_data, _ = librosa.effects.trim(audio_data)
            audio_data = audio_data[:661500]
            collection = np.split(audio_data,10)
            audio_data = collection[0]
            d = librosa.feature.mfcc(y=np.array(audio_data).flatten(),sr=22050 , n_mfcc = 20) #36565
            d_var = d.var(axis=1).tolist()
            d_mean = d.mean(axis=1).tolist()
            test_data = []#[d_var + d_mean]
            for i in range(20):
                test_data.append(d_mean[i])
                test_data.append(d_var[i])
                mfcc_names=[]
            for i in range(1,21):
                mfcc_str = "mfcc"+str(i)+"_mean"
                mfcc_names.append(mfcc_str)
                mfcc_str = "mfcc"+str(i)+"_var"
                mfcc_names.append(mfcc_str)
            test_frame = pd.DataFrame([test_data], columns = mfcc_names)
            test_data = []
            mfcc_names=[]
            #chroma
            S = np.abs(librosa.stft(audio_data, n_fft=4096))**2
            chroma = librosa.feature.chroma_stft(S=S, sr=sr)
            #chroma_stft_mean
            chroma_mean = round(np.mean(chroma),6)
            test_data.append(chroma_mean)
            #chrome_stft_var
            chroma_var = round(np.var(chroma),6)
            test_data.append(chroma_var)
            #chroma_label
            mfcc_names.append("chroma_stft_mean")
            mfcc_names.append("chroma_stft_var")

            #rms
            rms = librosa.feature.rms(y=audio_data)
            #rms_mean
            rms_mean = round(np.mean(rms),6)
            test_data.append(rms_mean)
            #rms_var
            rms_var = round(np.var(rms),6)
            test_data.append(rms_var)
            #rms_label
            mfcc_names.append("rms_mean")
            mfcc_names.append("rms_var")

            #spectral_centroid
            cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            #spectral_centroid_mean
            sc_mean = round(np.mean(cent),6)
            test_data.append(sc_mean)
            #spectral_centroid_var
            sc_var = round(np.var(cent),6)
            test_data.append(sc_var)
            #sc_label
            mfcc_names.append("spectral_centroid_mean")
            mfcc_names.append("spectral_centroid_var")

            #spectral_bandwidth
            spec_bw = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
            #spectral_bandwidth_mean
            spec_bw_mean = round(np.mean(spec_bw),6)
            test_data.append(spec_bw_mean)
            #spectral_bandwidth_var
            spec_bw_var = round(np.var(spec_bw),6)
            test_data.append(spec_bw_var)
            #sb_label
            mfcc_names.append("spectral_bandwidth_mean")
            mfcc_names.append("spectral_bandwidth_var")

            #rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
            #rolloff_mean
            rolloff_mean = round(np.mean(rolloff),6)
            test_data.append(rolloff_mean)
            #rolloff_var
            rolloff_var = round(np.var(rolloff),6)
            test_data.append(rolloff_var)
            #rolloff_label
            mfcc_names.append("rolloff_mean")
            mfcc_names.append("rolloff_var")

            #zero_crossing_rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            #zero_crossing_rate_mean
            zcr_mean = round(np.mean(zcr),6)
            test_data.append(zcr_mean)
            #zero_crossing_rate_var
            zcr_var = round(np.var(zcr),6)
            test_data.append(zcr_var)
            #zero_crossing_rate_label
            mfcc_names.append("zero_crossing_rate_mean")
            mfcc_names.append("zero_crossing_rate_var")

            #perceptr_mean
            #perceptr_var

            #tempo
            hop_length = 512
            oenv = librosa.onset.onset_strength(y=audio_data, sr=sr, hop_length=hop_length)
            tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                                    hop_length=hop_length)[0]

            tempo = round(tempo,6)
            test_data.append(tempo)
            #tempo_label
            mfcc_names.append("tempo")
            d_var = d.var(axis=1).tolist()
            d_mean = d.mean(axis=1).tolist()
            #test_data = []#[d_var + d_mean]
            for i in range(20):
                test_data.append(d_mean[i])
                test_data.append(d_var[i])
            for i in range(1,21):
                mfcc_str = "mfcc"+str(i)+"_mean"
                mfcc_names.append(mfcc_str)
                mfcc_str = "mfcc"+str(i)+"_var"
                mfcc_names.append(mfcc_str)

            scaler = pickle.load(open("/content/drive/MyDrive/projet_machine/ML/Machine_L/2 models train/scalar.pkl", 'rb'))
            X_train = pickle.load(open("/content/drive/MyDrive/projet_machine/ML/Machine_L/2 models train/xtrain.pkl", 'rb'))
            perm_features=pickle.load(open("/content/drive/MyDrive/projet_machine/ML/Machine_L/2 models train/perm_features.pkl",'rb'))

            # Assuming mfcc_names is defined correctly
            test_frame = pd.DataFrame([test_data], columns=mfcc_names)
            # Reorder the columns of test_frame to match X_train.columns
            test_frame = test_frame[X_train.columns]
            # Now you can apply the scaler
            testing_frame = pd.DataFrame(scaler.transform(test_frame), columns=X_train.columns)
            shorter_testing_frame = testing_frame[perm_features[:30]]

            val=1
            while(val<=9):
                audio_data = collection[val]
                d = librosa.feature.mfcc(y=np.array(audio_data).flatten(),sr=22050 , n_mfcc = 20) #36565
                d_var = d.var(axis=1).tolist()
                d_mean = d.mean(axis=1).tolist()
                test_data = []#[d_var + d_mean]
                for i in range(20):
                    test_data.append(d_mean[i])
                    test_data.append(d_var[i])
                mfcc_names=[]
                for i in range(1,21):
                    mfcc_str = "mfcc"+str(i)+"_mean"
                    mfcc_names.append(mfcc_str)
                    mfcc_str = "mfcc"+str(i)+"_var"
                    mfcc_names.append(mfcc_str)
                test_frame = pd.DataFrame([test_data], columns = mfcc_names)
                test_data = []
                mfcc_names=[]
                #chroma
                S = np.abs(librosa.stft(audio_data, n_fft=4096))**2
                chroma = librosa.feature.chroma_stft(S=S, sr=sr)
                #chroma_stft_mean
                chroma_mean = round(np.mean(chroma),6)
                test_data.append(chroma_mean)
                #chrome_stft_var
                chroma_var = round(np.var(chroma),6)
                test_data.append(chroma_var)
                #chroma_label
                mfcc_names.append("chroma_stft_mean")
                mfcc_names.append("chroma_stft_var")
              

                #rms
                rms = librosa.feature.rms(y=audio_data)
                #rms_mean
                rms_mean = round(np.mean(rms),6)
                test_data.append(rms_mean)
                #rms_var
                rms_var = round(np.var(rms),6)
                test_data.append(rms_var)
                #rms_label
                mfcc_names.append("rms_mean")
                mfcc_names.append("rms_var")

                #spectral_centroid
                cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
                #spectral_centroid_mean
                sc_mean = round(np.mean(cent),6)
                test_data.append(sc_mean)
                #spectral_centroid_var
                sc_var = round(np.var(cent),6)
                test_data.append(sc_var)
                #sc_label
                mfcc_names.append("spectral_centroid_mean")
                mfcc_names.append("spectral_centroid_var")

                #spectral_bandwidth
                spec_bw = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
                #spectral_bandwidth_mean
                spec_bw_mean = round(np.mean(spec_bw),6)
                test_data.append(spec_bw_mean)
                #spectral_bandwidth_var
                spec_bw_var = round(np.var(spec_bw),6)
                test_data.append(spec_bw_var)
                #sb_label
                mfcc_names.append("spectral_bandwidth_mean")
                mfcc_names.append("spectral_bandwidth_var")

                #rolloff
                rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
                #rolloff_mean
                rolloff_mean = round(np.mean(rolloff),6)
                test_data.append(rolloff_mean)
                #rolloff_var
                rolloff_var = round(np.var(rolloff),6)
                test_data.append(rolloff_var)
                #rolloff_label
                mfcc_names.append("rolloff_mean")
                mfcc_names.append("rolloff_var")

                #zero_crossing_rate
                zcr = librosa.feature.zero_crossing_rate(audio_data)
                #zero_crossing_rate_mean
                zcr_mean = round(np.mean(zcr),6)
                test_data.append(zcr_mean)
                #zero_crossing_rate_var
                zcr_var = round(np.var(zcr),6)
                test_data.append(zcr_var)
                #zero_crossing_rate_label
                mfcc_names.append("zero_crossing_rate_mean")
                mfcc_names.append("zero_crossing_rate_var")
                #perceptr_mean
                #perceptr_var
                #tempo
                hop_length = 512
                oenv = librosa.onset.onset_strength(y=audio_data, sr=sr, hop_length=hop_length)
                tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                                            hop_length=hop_length)[0]
                tempo = round(tempo,6)
                test_data.append(tempo)
                #tempo_label
                mfcc_names.append("tempo")
                d_var = d.var(axis=1).tolist()
                d_mean = d.mean(axis=1).tolist()
                #test_data = []#[d_var + d_mean]
                for i in range(20):
                    test_data.append(d_mean[i])
                    test_data.append(d_var[i])
                for i in range(1,21):
                    mfcc_str = "mfcc"+str(i)+"_mean"
                    mfcc_names.append(mfcc_str)
                    mfcc_str = "mfcc"+str(i)+"_var"
                    mfcc_names.append(mfcc_str)

                # Assuming mfcc_names is defined correctly
                test_frame2 = pd.DataFrame([test_data], columns=mfcc_names)
                # Reorder the columns of test_frame to match X_train.columns
                test_frame2 = test_frame2[X_train.columns]
                # Now you can apply the scaler
                testing_frame2 = pd.DataFrame(scaler.transform(test_frame2), columns=X_train.columns)
                shorter_testing_frame2 = testing_frame2[perm_features[:30]]
                df_test = pd.concat([shorter_testing_frame, shorter_testing_frame2])
                shorter_testing_frame = df_test
                val+=1


            rfc = pickle.load(open('/content/drive/MyDrive/projet_machine/ML/Machine_L/2 models train/Random Forest .pkl', 'rb'))
            cbc = pickle.load(open('/content/drive/MyDrive/projet_machine/ML/Machine_L/2 models train/CatBoost.pkl', 'rb'))
            xgbc = pickle.load(open('/content/drive/MyDrive/projet_machine/ML/Machine_L/2 models train/XGBoost.pkl', 'rb'))
            gbc = pickle.load(open('/content/drive/MyDrive/projet_machine/ML/Machine_L/2 models train/Gradient Boosting.pkl', 'rb'))
            abc = pickle.load(open('/content/drive/MyDrive/projet_machine/ML/Machine_L/2 models train/AdaBoost.pkl', 'rb'))
            lr = pickle.load(open('/content/drive/MyDrive/projet_machine/ML/Machine_L/2 models train/Logistic Regression.pkl', 'rb'))
            cls = pickle.load(open('/content/drive/MyDrive/projet_machine/ML/Machine_L/2 models train/KNN.pkl', 'rb'))

            #Testing Input Data
            #from collections import Counter
            result=[]
            models = {'Random Forest':rfc, 'Catboost':cbc, 'XGBoost':xgbc, 'Gradient Boosting':gbc, 'AdaBoost':abc,  'Logistic Regression':lr, 'KNN':cls}
            key_list = list(models.keys())
            val_list = list(models.values())


            for model_name, model in models.items():
              model_results = []  # Store predictions for the current model
              # Iterate over each sample in the test dataset
              for i in range(len(df_test)):
                test = model.predict(df_test[i:(i+1)])
                model_results.append(test)

              # Find the most common prediction in the results
              t = max(model_results, key=model_results.count)

              # Classify the most common prediction into genres
              if t == [[0]] or t == [['cha3ebi']]:
                    genre_detected = 'CHAABI'
              elif t == [[1]] or t == [['charki']]:
                    genre_detected = 'CHARKI'
              elif t == [[2]] or t == [['ray']]:
                    genre_detected = 'RAI'
              elif t == [[3]] or t == [['gnawa']]:
                    genre_detected = 'GNAWA'
              elif t == [[4]] or t == [['rap']]:
                    genre_detected = 'RAP'
              elif t == [[5]] or t == [['tachlhit']]:
                    genre_detected = 'TACHLHIT'
              elif t == [[6]] or t == [['ta9to9a']]:
                    genre_detected = 'TAKTOKA'
              else:
                    genre_detected = 'country'

              # Append the model name and its corresponding prediction to the result list
              result.append({'Model': model_name, 'Prediction': genre_detected})

            # Return the list of results
            return result




