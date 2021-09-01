import flask
import pickle
import pandas as pd
import numpy as np
from model.Predmodel import stacked_averaged_models
from scipy.stats import norm, skew #for some statistics
from scipy.special import boxcox1p

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
model = stacked_averaged_models
app = flask.Flask(__name__, template_folder='templates')

def home(input_variables):
    numeric_feats = input_variables.dtypes[input_variables.dtypes != "object"].index
    skewed_feats = input_variables[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness = skewness[abs(skewness) > 0.75]
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        input_variables[feat] = boxcox1p(input_variables[feat], lam)    
    return(input_variables)


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
       return(flask.render_template('index.html')) 
    if flask.request.method == 'POST':
       LA = flask.request.form['LA']
       FP = flask.request.form['FP']
       OQ = flask.request.form['OQ']
       SqFt = flask.request.form['SqFt']
       GSqFt = flask.request.form['GSqFt']
       Bath = flask.request.form['Bath']
       CG = flask.request.form['CG']
       TatSqFt = flask.request.form['TatSqFt']
       Rms = flask.request.form['Rms']
       Year = flask.request.form['Year']
       input_variables = pd.DataFrame([[LA,FP,OQ,SqFt,GSqFt,Bath,CG,TatSqFt,Rms,Year]],columns = ['LotArea', 'Fireplaces','OverallQual','GrLivArea','GarageArea','FullBath','GarageCars','TotalSF','TotRmsAbvGrd','YearBuil'],dtype=int)
       home(input_variables)
       prediction = model.predict(input_variables)[0]
       prediction = np.round(prediction,decimals=2)
       return flask.render_template('index.html',original_input={'LotArea':LA,'Fireplaces':FP,'OverallQual':OQ,'GrLivArea':SqFt,'GarageArea':GSqFt,'FullBath':Bath,'GarageCars':CG,'TotalSF':TatSqFt,'TotRmsAbvGrd':Rms,'YearBuil':Year},result=prediction)
if __name__ == '__main__':
   app.run()