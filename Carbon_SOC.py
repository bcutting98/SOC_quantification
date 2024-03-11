# -*- coding: utf-8 -*-

#File path for in and out files

Input_path = "C:\\Users\\bc00838\\OneDrive - University of Surrey\\Documents\\3. Python Code\\Data dump\\"
Out_path = ""

#Import modules
import numpy as np
import pandas as pd
import joblib
import optuna
import copy
import time

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as sklm
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

Max_features = 0
seed = 0 #IMPORTANT seed for split and RF/SVR process         

def ratio_dif(a,b,c=0):
    return (a-b)/(a+b+c)

#Add spectral indices

def add_indices(Df):
    Df['RVI'] = Df['B8']/Df['B4'] 
    Df['NDCI'] = ratio_dif(Df['B7'],Df['B4']) 
    Df['GNDVI'] = ratio_dif(Df['B8'],Df['B3']) 
    Df['OSAVI'] = 1.16*(ratio_dif(Df['B8'],Df['B4'],0.16)) 
    Df['NDRE'] = ratio_dif(Df['B8'],Df['B5']) 
    Df['CARI'] = ((Df['B5'] - Df['B4'])/(0.2*(Df['B5'] + Df['B4'])))*(Df['B5']/Df['B4']) 
    Df['TVI'] = 0.5*((120*(Df['B6'] - Df['B3'])) - 200*(Df['B4'] - Df['B3'])) 
    Df['MTVI'] = 1.2*(1.2*(Df['B8'] - Df['B3'])-2.5*(Df['B4'] - Df['B3'])) 
    Df['EVI'] = 2.5*((Df['B8'] - Df['B4'])/(Df['B8'] + 6*Df['B4'] - 7.5 * Df['B2'] + 0.5)) 
    Df['NVI'] = (Df['B7'] - Df['B6'])/Df['B4'] 
    Df['MSAVI'] = 0.5*(2*Df['B8'] + 1 - np.sqrt((2*Df['B8'] + 1)**2 - 8*(Df['B8'] - Df['B4'])))


#Define SVR optimisation for optuna

def objective_SVR(trial):    
    
    joblib.dump(study, 'study_SVR.pkl')
    pca__n_components = trial.suggest_int('pca__n_components',2,Max_features)
    svr__C = 10**trial.suggest_float('svr__C', -1, 2) 
    svr__gamma = 10**trial.suggest_float('svr__gamma', -3, 2) 
    svr__epsilon = 10**trial.suggest_float('svr__epsilon', -4, 2) 
    model = Pipeline([('standardscaler', StandardScaler()), ('pca', PCA(pca__n_components)), ('svr', SVR(kernel='rbf',C=svr__C,epsilon=svr__epsilon,gamma=svr__gamma))])


    params = {
        'svr__C': svr__C,
        'svr__gamma': svr__gamma,
        'svr__epsilon': svr__epsilon,
        'pca__n_components': pca__n_components
       
    }
    
    model.set_params(**params)
    cv = sklm.KFold(n_splits=fold_no,shuffle=True,random_state=seed)
    rscores = cross_val_score(model, Feature_train, SOC_train, cv=cv, n_jobs=-1,scoring='neg_mean_squared_error')
    return  (-np.mean(rscores))


#Define RF optimisation for optuna

def objective_RF(trial):    
    
    joblib.dump(study, 'study_RF.pkl')
    pca__n_components = trial.suggest_int('pca__n_components',1,Max_features)
    RF__n_estimators = trial.suggest_int('rf__num_estimators', 500, 25000) 
    RF__max_depth = trial.suggest_int('rf__max_depth', 5, 4000) 
    RF__max_features = trial.suggest_float('rf__max_features',0.1,1)

    params = {
        'pca__n_components': pca__n_components,
        'RF__n_estimators': RF__n_estimators,
        'RF__max_depth': RF__max_depth,
        'RF__max_features': RF__max_features
       
    }
        
    model = Pipeline([('transformer', StandardScaler()), ('pca',PCA(pca__n_components)), ('RF', RandomForestRegressor(n_estimators=RF__n_estimators,max_features=RF__max_features,max_depth=RF__max_depth,random_state = 14,n_jobs=-1))])
    
    model.set_params(**params)
    cv = sklm.KFold(n_splits=fold_no,shuffle=True,random_state=seed)
    rscores = cross_val_score(model, Feature_train, SOC_train, cv=cv, n_jobs=-1,scoring='neg_mean_squared_error')
    return  (-np.mean(rscores))

#Basic SVR pipeline returning model stats
def SVR_pipe(Features, y_values, regularise, eps, comps, folds, seed=seed, gamma=0, fit=True):#

    scalar = StandardScaler()
    if gamma == 0:
        svr_rbf = SVR(kernel="rbf", C=regularise, epsilon=eps,gamma='scale')
    else:
        svr_rbf = SVR(kernel="rbf", C=regularise, epsilon=eps, gamma=gamma)
    pipeline = Pipeline([('transformer', scalar), ('pca',PCA(n_components=comps)), ('estimator', svr_rbf)])
    cv = sklm.KFold(n_splits=folds,shuffle=True,random_state=seed)
    rmses = cross_val_score(pipeline, Features, y_values,  cv = cv,scoring = "neg_mean_squared_error")
    r2s = cross_val_score(pipeline, Features, y_values,  cv = cv,scoring = "r2",n_jobs=-1)
    maes = cross_val_score(pipeline, Features, y_values,  cv = cv,scoring = "neg_mean_absolute_error",n_jobs=-1)
    scores = [rmses,r2s,maes]
    if fit == True:
        SOC_pred = cross_val_predict(pipeline, Features, y_values,  cv = cv)
        pipeline.fit(Features,y_values)
        r = permutation_importance(pipeline, Features, y_values,
        n_repeats=30, 
        random_state=seed,
        scoring="neg_mean_squared_error")
        return scores, SOC_pred, r, pipeline
    else:
        return scores
 
    
 #Basic RF pipeline returning model stats
def RF_pipe(Features,y_values, Estimators, M_features, comp, M_depth=None, bar_labels=None,seed=12,folds=5,fit=True):
    scalar = StandardScaler()
    pipeline = Pipeline([('transformer', scalar), ('pca',PCA(n_components=comp)), ('RF', RandomForestRegressor(n_estimators=Estimators,max_features=M_features,max_depth=M_depth,random_state = seed,n_jobs=-1))])
    cv = sklm.KFold(n_splits=folds,shuffle=True,random_state=seed)
    rmses = cross_val_score(pipeline, Features, y_values,  cv = cv,scoring = "neg_mean_squared_error")
    r2s = cross_val_score(pipeline, Features, y_values,  cv = cv,scoring = "r2",n_jobs=-1)
    maes = cross_val_score(pipeline, Features, y_values,  cv = cv,scoring = "neg_mean_absolute_error",n_jobs=-1)
    scores = [rmses,r2s,maes]
    if fit==True:
        SOC_pred = cross_val_predict(pipeline, Features, y_values,  cv = cv)
        pipeline.fit(Features,y_values)
        r = permutation_importance(pipeline, Features, y_values,
        n_repeats=30,
        random_state=seed,
        scoring="neg_mean_squared_error",
        n_jobs=-1)
        return scores,SOC_pred,r, pipeline
    else:
        return scores
    
        

if __name__ == "__main__":
    
#############################################
    
    model_type = 0 #SVR = 0, RF = 1
    iterations = 500
    fold_no = 5
    conv_factor = .55 #Convert SOC to SOM

    add_spec_ind = True #Spectral indices
    add_topo = True #Topographical indices
    extra_bands = False #Low res bands (B1+B9)
    single_date = False #Single or multi date
    save_data = True #Save results
    
    
 ############################################ 


    # Clean and filter data
        
    S2_data_2021 = pd.read_csv(Input_path + 'SOC.csv')
    S2_data_2021 = S2_data_2021[['ID','SOC']]
    S2_data_2021.rename(columns={'ID':'Sample'}, inplace=True)
        
    Temporal_data = pd.read_csv(Input_path + 'Filtered_bands.csv')
    Lidar = pd.read_csv(Input_path + 'Topography_trim.csv')
    Lidar['Slope'] = Lidar['Slope']/100 
        
    if extra_bands == True:
        clean = ['SCL','Sample','year','DOY']
        bands = 12
    else:
        clean = ['SCL','Sample','year','DOY','B1','B9']
        bands = 10
    
    if add_topo == True:
        Temporal_data = Temporal_data.merge(Lidar,on='Sample',how='left')
    
    Temporal_data_SOC = Temporal_data.merge(S2_data_2021,on='Sample',how='left')
    
    
    Temporal_data_fil = Temporal_data_SOC[Temporal_data_SOC['year'].isin([2020,2021,2022,2023]) & Temporal_data_SOC['SCL'].isin([4,5,7])]
    Temporal_data_fil = Temporal_data_fil[(Temporal_data_fil['DOY'] > 69) | (Temporal_data_fil['year'] != 2020)]
    
            
    if add_spec_ind == True:    
        add_indices(Temporal_data_fil)  

    Temporal_data_fil = Temporal_data_fil[[c for c in Temporal_data_fil if c not in ['SOC']] 
                                                  + ['SOC']]

    unique_points = Temporal_data_fil[['Sample','SOC']]
    unique_points = unique_points.drop_duplicates()
    unique_points = unique_points.dropna(axis=0)
    unique_points = unique_points.to_numpy()
    unique_points = unique_points[unique_points[:, 0].argsort()]
    Temporal_data_fil_II = Temporal_data_fil.dropna(axis=0)
    Total  = Temporal_data_fil_II.shape[0]
    
    if single_date == True:

        state1 = Temporal_data_fil_II['DOY']==251
        state2 = Temporal_data_fil_II['year']==2021
        Temporal_data_fil_III  = Temporal_data_fil_II[state1 & state2]
        
    else:
        Temporal_data_fil_III = Temporal_data_fil_II.groupby('Sample').median()
        add_indices(Temporal_data_fil_III)
        clean.remove('Sample')
    
    S2_features = Temporal_data_fil_III.drop(labels=clean,axis=1)
    bar_labels = np.delete(np.array(S2_features.columns.values).astype(str),-1)
    S2_features_num = S2_features.to_numpy()
    Features, SOC = np.delete(S2_features_num,-1,axis=1),S2_features_num[:,-1]
    SOC = conv_factor*SOC
    Features = Features.astype(float)
    Max_features = Features.shape[1]
    Temp = Features[:,0:bands]
    Temp = -np.log10(Temp)
    Features[:,0:bands] = Temp
    
    #Split data for testing and training
    Feature_train, Feature_test, SOC_train, SOC_test = train_test_split(Features, SOC, test_size=0.2, random_state=seed)

    study = optuna.create_study()


    if model_type == 0:
        study.optimize(objective_SVR, n_trials=iterations)
        joblib.dump(study, 'study_SVR.pkl')
    else:
        study.optimize(objective_RF, n_trials=iterations)
        joblib.dump(study, 'study_RF.pkl')


            
            
    #Return best model parameters and insert to pipe
    # to record the value for the last time
    best = study.best_params
    if model_type == 0:
        scores,SOC_sim,importance, pipeline = SVR_pipe(Feature_train, SOC_train, comps=best['pca__n_components'], folds=fold_no,regularise=10**best['svr__C'],seed=12,eps=10**best['svr__epsilon'],gamma=10**best['svr__gamma'])
    else:
        scores, SOC_sim, importance, pipeline = RF_pipe(Feature_train,SOC_train,best['rf__num_estimators'],comp=best['pca__n_components'],M_features = best['rf__max_features'],M_depth=best['rf__max_depth'],bar_labels=bar_labels,folds=fold_no)
    
    cv_rmse_avg = [np.sqrt(-np.mean(scores[0])),np.mean(scores[1]),-np.mean(scores[2])]
    cv_rmse_std = [0.5*np.std(scores[0]),np.std(scores[1]),np.std(scores[2])]
    
    #Print and save evaluation stats
    #Note importances and stored in the importance variable
    print("Cross Validation")
    print("RMSE = %0.4f +/- %0.4f" % (np.sqrt(-np.mean(scores[0])),0.5*np.std(scores[0])))
    print("R2 = %0.4f +/- %0.4f" % (np.mean(scores[1]),np.std(scores[1])))
    print("MAE = %0.4f +/- %0.4f" % (-np.mean(scores[2]),np.std(scores[2])))
            
    
    
    SOC_sim_II = pipeline.predict(Feature_test)
    
    val_rmse = [mean_squared_error(SOC_test,SOC_sim_II,squared=False),r2_score(SOC_test,SOC_sim_II),mean_absolute_error(SOC_test,SOC_sim_II)]


    print("Validation")
    print("RMSE = %0.4f" % mean_squared_error(SOC_test,SOC_sim_II,squared=False))
    print("R2 = %0.4f" % r2_score(SOC_test,SOC_sim_II))
    print("MAE = %0.4f" % mean_absolute_error(SOC_test,SOC_sim_II))

    results = pd.DataFrame([cv_rmse_avg,cv_rmse_std,val_rmse],columns=['RMSE','R2','MAE'])
    results.index = ['CV','CV_std','Val']
    if save_data == True:
        results.to_csv(Out_path + 'results' + str(time.strftime("%Y%m%d-%H%M%S")))

