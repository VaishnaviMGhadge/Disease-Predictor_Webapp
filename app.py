import pickle
from flask import Flask,render_template,request
import numpy as np
import pandas as pd

app = Flask(__name__, static_url_path='/static') 


@app.route('/',methods=['GET','POST'])
def home():
    return render_template('home.html')

@app.route('/cancer',methods=['GET','POST'])
def cancer():
    if request.method=='GET':
        return render_template('form.html')
    else:
        with open('model_pickle','rb') as f:
            mp=pickle.load(f)
        df=pd.read_csv('wisc_bc_data.csv')
        df['diagnosis'].replace({'M':1,'B':0},inplace=True) 
        radius_mean=float(request.form['radius_mean'])
        area_mean=float(request.form['area_mean'])
        perimeter_mean=float(request.form['perimeter_mean'])
        area_worst=float(request.form['area_worst'])
        perimeter_worst=float(request.form['perimeter_worst'])
        area_se=float(request.form['area_se'])
        
        new=np.array([[df['id'].mean(),radius_mean,df['texture_mean'].mean(),perimeter_mean,area_mean,df['smoothness_mean'].mean(),df['compactness_mean'].mean(),df['concavity_mean'].mean(),df['concave points_mean'].mean(),df['symmetry_mean'].mean(),df['fractal_dimension_mean'].mean(),df['radius_se'].mean(),df['texture_se'].mean(),df['perimeter_se'].mean(),area_se,df['smoothness_se'].mean(),df['concavity_se'].mean(),df['concave points_se'].mean(),df['symmetry_se'].mean(),df['fractal_dimension_se'].mean(),df['radius_worst'].mean(),df['texture_worst'].mean(),perimeter_worst,area_worst,df['smoothness_worst'].mean(),df['compactness_worst'].mean(),df['concavity_worst'].mean(),df['concave points_worst'].mean(),df['symmetry_worst'].mean(),df['fractal_dimension_worst'].mean()]])
        y_pred=mp.predict(new)
        return render_template('result.html',y_pred=y_pred)

        
@app.route('/heart',methods=['GET','POST'])
def HeartDisease():
    if request.method=='GET':
        return render_template('form1.html')
    else:
        with open('reg_pickle','rb') as t:
            mp1=pickle.load(t)
        df=pd.read_csv('datasets_4123_6408_framingham.csv')
        
        male=int(request.form['male'])
        age=int(request.form['age'])
        cigsPerDay=int(request.form['cigsPerDay'])
        totChol=int(request.form['totChol'])
        sysBP=int(request.form['sysBP'])
        diaBP=int(request.form['diaBP'])
        glucose=int(request.form['glucose'])
        new1=np.array([[male,age,df['currentSmoker'].mean(),cigsPerDay,df['BPMeds'].mean(),df['prevalentStroke'].mean(),df['prevalentHyp'].mean(),df['diabetes'].mean(),totChol,sysBP,diaBP,df['BMI'].mean(),df['heartRate'].mean(),glucose]])
        y_pred=mp1.predict(new1)
        return render_template('result1.html',y_pred=y_pred)


@app.route('/liver',methods=['GET','POST'])
def liver():
    if request.method=='GET':
        return render_template('form3.html')
    else:
        with open('model2_pickle','rb') as f:
            mp3=pickle.load(f)
        df=pd.read_csv('indian_liver_patient.csv')
        age=int(request.form['age'])
        Gender=str(request.form['Gender'])
        Total_Bilirubin=float(request.form['Total_Bilirubin'])
        Direct_Bilirubin=float(request.form['Direct_Bilirubin'])
        Alkaline_Phosphotase=int(request.form['Alkaline_Phosphotase'])
        Alamine_Aminotransferase=int(request.form['Alamine_Aminotransferase'])
        #Aspartate_Aminotransferase=int(request.form['Aspartate_Aminotransferase'])
        new3=np.array([[age,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,df['Aspartate_Aminotransferase'].mean()]])
        y_pred=mp3.predict(new3)
        return render_template('result3.html',y_pred=y_pred)
        #return f'{y_pred}'

if __name__=='__main__':
    app.run(debug=True)













