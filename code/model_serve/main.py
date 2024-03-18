import functions_framework
import os
import io
#import pickle
#import json
import joblib
import glob
import librosa
from gcloud import storage
import numpy as np
from flask import Flask, jsonify
from pydub import AudioSegment
from pydub.utils import make_chunks
from tempfile import TemporaryFile

PROJECT_ID = 'wellsfargo-genai24-8038'
model_file_name="svm/svm_model.pkl"
bucket_name = "auracle_models"
scaler_file_name="svm/svm_scaler.pkl"
REGION = 'us-east4'
ARTIFACT_URI=f"gs://{bucket_name}"
BLOB_NAME =  model_file_name

   

def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfccs.T, axis=0)

@functions_framework.http
def hello_http(request):

    request_json = request.get_json(silent=True)
    request_args = request.args

    storage_client = storage.Client()
    # Create a bucket object for our bucket
    bucket = storage_client.get_bucket(bucket_name)
    # Create a blob object from the filepath
    blob = bucket.blob(model_file_name)
    with TemporaryFile() as temp_file:
        #download blob into temp file
        blob.download_to_file(temp_file)
        temp_file.seek(0)
        #load into joblib
        model=joblib.load(temp_file)


    blob1 = bucket.blob(scaler_file_name)
    with TemporaryFile() as temp_file_sclr:
        #download blob into temp file
        blob1.download_to_file(temp_file_sclr)
        temp_file_sclr.seek(0)
        #load into joblib
        scaler=joblib.load(temp_file_sclr)


    file =  request.files['file']
    print(file)
    #INSERTED FROM HERE
    audio_bytes_mp3 = file.read()    
    testiomp3=io.BytesIO(audio_bytes_mp3)
    mp3AudioSegement = AudioSegment.from_file(testiomp3)

    with TemporaryFile() as temp_file_wave:
        mp3AudioSegement.export(temp_file_wave)
        temp_file_wave.seek(0)
        myaudio = AudioSegment.from_file(io.BytesIO(temp_file_wave.read())) 

    #save_path = os.path.join(os.getcwd(),"/temp.mp3")
    #file.save(save_path)
    #request.files['messageFile'].save(save_path)

    #wav_output_file=os.path.join(os.getcwd(),"/temp.wav")

    #sound = AudioSegment.from_file(save_path)
    #sound.export(wav_output_file, format="wav")

    chunk_length_ms = 2000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of two sec
    exported_chunk = ''

    print(str(len(chunks)))

    #Export all of the individual chunks as wav files
    for i, chunk in enumerate(chunks):
        chunk_name = "chunk{0}.wav".format(i)
        if len(chunks) == 1 :
           with TemporaryFile() as temp_chunk_file:
            chunk.export(temp_chunk_file,format="wav")
            exported_chunk=temp_chunk_file
            break
        elif len(chunks) > 1 and i == 2:
           with TemporaryFile() as temp_chunk_file:
            chunk.export(temp_chunk_file,format="wav")
            exported_chunk=temp_chunk_file
            #chunk.export(chunk_name, format="wav")
        elif i > 2: 
            break


    mfcc_features = extract_mfcc_features(exported_chunk)
    svcprb_classifier = model
    setflag=''
    result=''
    aiprob=''
    hprob=''
    if mfcc_features is not None:
        mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))
        prediction = svcprb_classifier.predict(mfcc_features_scaled)
        probs= svcprb_classifier.predict_proba(mfcc_features_scaled)
        classifier = ''
        if prediction[0] == 1:
            result = 'human'
            setflag = 'true'
            print("Classified : Human")
            hprob = str("{:.2f}%".probs[0][1]*100)
            print("Human Probability score: {:.0f}%".format(probs[0][1]*100))
        else:
            result = 'ai'
            setflag = 'false'
            aiprob = str("{:.2f}%".probs[0][1]*100)
            print("Classified : AI")
            print("AI Probability score: {:.0f}%".format(probs[0][0]*100))
    else:
        print("Error: Unable to process the input audio.")
        
    output ={
            "status": "success",
                "analysis": {
                    "detectedVoice": setflag,
                    "voiceType": result,
                "confidenceScore": {
                "aiProbability": aiprob, 
                "humanProbability": hprob
                } ,
                "additionalInfo": {
                "emotionalTone": "neutral",
                "backgroundNoiseLevel":"low"
                },
                "responseTime": 200
            }
        }  
    return jsonify(output)