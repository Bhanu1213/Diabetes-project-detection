import numpy as np
from flask import Flask,request,render_template
import pickle as pk

app=Flask(__name__)
model=pk.load(open('classifier.pkl','rb'))
sc=pk.load(open('sc.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Result', methods=['POST'])
def Result():
        result1 = [float(request.form['Age']), float(request.form['BMI']),
                   float(request.form['BloodPressure']), float(request.form['Insulin'])]
        final_result = [np.array(result1)]
        final_result = sc.transform(final_result)
        result = model.predict(final_result)
        return render_template('output.html', prediction=result[0])  
    

if __name__=="__main__":
    app.run(debug=True)