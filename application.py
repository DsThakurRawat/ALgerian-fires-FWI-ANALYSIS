from flask import Flask, render_template ,jsonify,request #type source .venv/bin/activatesource .venv/bin/activate
import pickle
import numpy as np  
import pandas as pd
application = Flask(__name__)
app = application

##import ridge regressor and standard scaler pickle
ridge_model = pickle.load(open("models/ridge.pkl","rb"))
standard_scaler  = pickle.load(open("models/sc.pkl","rb"))

#next we are going to create index

@app.route('/')
def index():
    return render_template("index.html")


@app.route("/predictdata",methods = ["GET","POST"])
#creating my definition
def predict_datapoint():

   if request.method == "POST":
       Temperature = float(request.form.get("Temperature"))
       RH = float(request.form.get("RH")) #these are input textbox
       Ws = float(request.form.get("Ws"))
       Rain = float(request.form.get("Rain"))
       FFMC = float(request.form.get("FFMC"))
       DMC = float(request.form.get("DMC"))
       ISI = float(request.form.get("ISI"))
     
       new_data_scaled = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI]])
       result = ridge_model.predict(new_data_scaled)
       
       return render_template("home.html",results = result[0])


   
   else:
       return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
    