import flask
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from scipy.special import boxcox1p
from scipy import stats
from scipy.stats import norm, skew #for some statistics

from model.Predmodel import stacked_averaged_models

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