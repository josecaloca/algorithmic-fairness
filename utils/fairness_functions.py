#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fairness_functions.py

Purpose:
    Performing all necessary calculations for master's dissertation as Warsaw School of Economics.

Date:
    16/06/2022

Author:
    Jose Caloca
"""
###########################################################
### Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms import preprocessing
from aif360.algorithms import postprocessing


###########################################################
### dY= emptyfunc(vX)
def LoadData(iObs, bPreSampled = True):
    """
    Purpose:
        Load in the data from a pre-specified .csv file containing the HMDA
        data for 2017, nationwide. Due to the size of this file, we provide a
        pre-sampled version zipped together with the data. the boolean bPreSampled
        notes whether the pre-sampled version of the data must be used, or if the
        entire file is sampled. Pre-sampled is done with iObs rows and all columns.
        These observations have been cleaned using the PrepData() function.
        
        The commented lines can be uncommented when the dataframes must be converted to
        .csv files. Furthermore, it also creates a .csv with the used rows from the
        dataframe. For the pre-sampled version, this .csv is provided.
        
        For the pre-sampled version, a .csv file called "SampledData.csv" is required,
        which is zipped together with this script.

    Inputs:
        iObs            Integer, number of observations used for sampling
        bPreSampled     Boolean, notes whether pre-sampled data must be used

    Return value:
        dfZsampled      iObs x iK dataframe, contains sampled data
    """
    if bPreSampled == False:
        dfZ = pd.read_csv("data\hmda_2017_nationwide_all-records_labels.zip")
        print("reading done")
        # mZ = np.array(dfZ)
        dfZprepped = PrepData(dfZ)
        
        vRand = random.sample(range(dfZprepped.shape[0]),iObs)
        mZsampled = []
        for i in range(iObs):
            mZsampled.append(dfZprepped.iloc[vRand[i]])
        print("starting creating the dataframe")
        dfZsampled = pd.DataFrame(mZsampled, columns=dfZprepped.columns)
        print("dataframe done, thank god")
        
        # # Following lines can be uncommented if sample is to be exported to .csv
        # dfZsampled.to_csv('data\sample.csv')
        
        # dfRandN = pd.DataFrame(vRand)           # Also for random numbers for clarity
        # dfRandN.to_csv('data\random_numbers.csv')
    
    if bPreSampled == True:
        dfZsampled = pd.read_csv("./data/sample.csv")
        dfZsampled = dfZsampled.drop('Unnamed: 0', axis=1)
        print("Dataframe succesfully loaded")
        # mZsampled = dfZsampled.to_numpy()
        
    
    return dfZsampled           # Add mZsampled if previous line is uncommented

###########################################################
### dY= emptyfunc(vX)
def PrepData(dfZ):
    """
    Purpose:
        Prepares the data for use, winsorizes columns and adds target column.
        
    Inputs:
        dfZ         iObs x iK dataframe, contains sampled data
        
    Return value:
        dfZq        iObs x (iK + 1) dataframe, contains prepped data
        
    """
    dfZ = dfZ[dfZ['action_taken']!=4]       # Remove loan withdrawels
    dfZ['TARGET']= dfZ['action_taken']==1   # Make approval a boolean
    
    dfSort = dfZ[dfZ['applicant_income_000s'] > 0]
    dfSort = dfSort[dfSort['loan_amount_000s'] > 0]
    
    dfZq = dfSort[dfSort['loan_amount_000s']<np.quantile(dfSort['loan_amount_000s'],0.99)]
    dfZq = dfZq[dfZq['applicant_income_000s']<np.quantile(dfZq['applicant_income_000s'],0.99)]
    # dfZqq = dfZq[dfZq['applicant_income_000s']<np.quantile(dfZq['applicant_income_000s'],0.99)]
    
    return dfZq         # one q removed       

###########################################################
### dY= emptyfunc(vX)
def DropData(dfZ):
    """
    Purpose:
        An extensive function to clean up the dataset to prepare it for input
        into machine learning models. Drop columns, fills NA-values and encodes
        the values.

    Inputs:
        dfZ         dataframe, contains the dataset which has gone through PrepData

    Return value:
        dfZdropped  dataframe, cleaned

    """
    dfZdropped = dfZ.copy(deep=True)
    dfZdropped.drop(['application_date_indicator', 'sequence_number', 'edit_status', 
                          'edit_status_name', 'lien_status', 'hoepa_status', 'denial_reason_3',
                          'denial_reason_2', 'denial_reason_1', 'purchaser_type', 'co_applicant_sex',
                          'applicant_sex', 'co_applicant_race_5', 'co_applicant_race_4',
                          'co_applicant_race_3', 'co_applicant_race_2', 'co_applicant_race_1',
                          'applicant_race_5', 'applicant_race_4', 'applicant_race_3',
                          'applicant_race_2', 'applicant_race_1', 'co_applicant_ethnicity',
                          'applicant_ethnicity', 'msamd', 'action_taken', 'preapproval',
                          'owner_occupancy', 'loan_purpose', 'property_type', 'loan_type'], 
                          axis=1,inplace=True)
    dfZdropped.drop(['as_of_year','co_applicant_race_name_2','co_applicant_race_name_3',
                                  'co_applicant_race_name_4','co_applicant_race_name_5','state_abbr',
                                  'state_code','rate_spread','applicant_race_name_5','applicant_race_name_4',
                                  'applicant_race_name_3','applicant_race_name_2','denial_reason_name_3',
                                  'denial_reason_name_2','denial_reason_name_1', 'county_code', 'applicant_ethnicity_name', 
                                  'census_tract_number', 'co_applicant_ethnicity_name', 'agency_name', 'agency_abbr'], 
                                  axis=1,inplace=True)


    dfZdropped['tract_to_msamd_income'].fillna(value=dfZdropped['tract_to_msamd_income'].mean(),inplace=True)
    dfZdropped['population'].fillna(value=dfZdropped['population'].mean(),inplace=True)
    dfZdropped['minority_population'].fillna(value=dfZdropped['minority_population'].mean(),inplace=True)
    dfZdropped['number_of_owner_occupied_units'].fillna(value=dfZdropped['number_of_owner_occupied_units'].mean(),inplace=True)
    dfZdropped['number_of_1_to_4_family_units'].fillna(value=dfZdropped['number_of_1_to_4_family_units'].mean(),inplace=True)
    dfZdropped['loan_amount_000s'].fillna(value=dfZdropped['loan_amount_000s'].mean(),inplace=True)
    dfZdropped['hud_median_family_income'].fillna(value=dfZdropped['hud_median_family_income'].mean(),inplace=True)
    dfZdropped['applicant_income_000s'].fillna(value=dfZdropped['applicant_income_000s'].mean(),inplace=True)
    dfZdropped['respondent_id'].fillna(value=dfZdropped['respondent_id'].mode(),inplace=True)
    dfZdropped['purchaser_type_name'].fillna(value=dfZdropped['purchaser_type_name'].mode(),inplace=True)
    dfZdropped['property_type_name'].fillna(value=dfZdropped['property_type_name'].mode(),inplace=True)
    dfZdropped['preapproval_name'].fillna(value=dfZdropped['preapproval_name'].mode(),inplace=True)
    dfZdropped['owner_occupancy_name'].fillna(value=dfZdropped['owner_occupancy_name'].mode(),inplace=True)
    dfZdropped['msamd_name'].fillna(value=dfZdropped['msamd_name'].mode(),inplace=True)
    dfZdropped['state_name'].fillna(value=dfZdropped['state_name'].mode(),inplace=True)
    dfZdropped['loan_type_name'].fillna(value=dfZdropped['loan_type_name'].mode(),inplace=True)
    dfZdropped['loan_purpose_name'].fillna(value=dfZdropped['loan_purpose_name'].mode(),inplace=True)
    dfZdropped['lien_status_name'].fillna(value=dfZdropped['lien_status_name'].mode(),inplace=True)
    dfZdropped['hoepa_status_name'].fillna(value=dfZdropped['hoepa_status_name'].mode(),inplace=True)
    dfZdropped['county_name'].fillna(value=dfZdropped['county_name'].mode(),inplace=True)
    dfZdropped['co_applicant_sex_name'].fillna(value=dfZdropped['co_applicant_sex_name'].mode(),inplace=True)
    dfZdropped['co_applicant_race_name_1'].fillna(value=dfZdropped['co_applicant_race_name_1'].mode(),inplace=True)
    dfZdropped['applicant_sex_name'].fillna(value=dfZdropped['applicant_sex_name'].mode(),inplace=True)
    dfZdropped['applicant_race_name_1'].fillna(value=dfZdropped['applicant_race_name_1'].mode(),inplace=True)
    dfZdropped['action_taken_name'].fillna(value=dfZdropped['action_taken_name'].mode(),inplace=True)
    
    labelencoder=LabelEncoder()
    dfZdropped['purchaser_type_name'] = labelencoder.fit_transform(dfZdropped['purchaser_type_name'])
    dfZdropped['property_type_name'] = labelencoder.fit_transform(dfZdropped['property_type_name'])
    dfZdropped['preapproval_name'] = labelencoder.fit_transform(dfZdropped['preapproval_name'])
    dfZdropped['owner_occupancy_name'] = labelencoder.fit_transform(dfZdropped['owner_occupancy_name'])
    dfZdropped['loan_type_name'] = labelencoder.fit_transform(dfZdropped['loan_type_name'])
    dfZdropped['loan_purpose_name'] = labelencoder.fit_transform(dfZdropped['loan_purpose_name'])
    dfZdropped['lien_status_name'] = labelencoder.fit_transform(dfZdropped['lien_status_name'])
    dfZdropped['hoepa_status_name'] = labelencoder.fit_transform(dfZdropped['hoepa_status_name'])
    dfZdropped['county_name'] = labelencoder.fit_transform(dfZdropped['county_name'].astype(str))
    dfZdropped['co_applicant_sex_name'] = labelencoder.fit_transform(dfZdropped['co_applicant_sex_name'])
    dfZdropped['co_applicant_race_name_1'] = labelencoder.fit_transform(dfZdropped['co_applicant_race_name_1'])
    dfZdropped['applicant_sex_name'] = labelencoder.fit_transform(dfZdropped['applicant_sex_name'])
    dfZdropped['applicant_race_name_1'] = labelencoder.fit_transform(dfZdropped['applicant_race_name_1'])
    
    dfZdropped['msamd_name'] = labelencoder.fit_transform(dfZdropped['msamd_name'].astype(str))
    dfZdropped['state_name'] = labelencoder.fit_transform(dfZdropped['state_name'].astype(str))
    
    dfZdropped = dfZdropped.drop(['TARGET'], axis=1)
    dfZdropped['TARGET']=["approved" if x=="Loan originated" else "not approved" for x in dfZdropped['action_taken_name']]
    dfZdropped = dfZdropped.drop(['action_taken_name'], axis=1)
    dfZdropped['TARGET'] = labelencoder.fit_transform(dfZdropped['TARGET'])
    
    dfZdropped = dfZdropped.drop_duplicates()
    
    dfZdropped['TARGET'] = dfZdropped['TARGET'] + 1
    dfZdropped['TARGET'][dfZdropped['TARGET'] == 2] = 0
    
    return dfZdropped        


###########################################################
### dY= emptyfunc(vX)
def TestFun(dfZ):
    """
    Purpose:
        Compare multiple variables belonging to the same group, 
        with t-testing and f-testing

    Inputs:
        dFZ          Dataframe, containing all the groups
        

    Return value:
        dfTest      Dataframe, containing results of the tests
    """
    # Dataframe to store everything
    dfTest = pd.DataFrame()
    # Initialisation male / female
    dF= dfZ[["TARGET","applicant_sex_name","applicant_race_name_1"]]


    dfFemales = dF[dF["applicant_sex_name"] == 0]
    dfMales = dF[dF["applicant_sex_name"] == 2]
    dfO = dF[(dF["applicant_sex_name"] != 2) & (dF["applicant_sex_name"] != 0)]
    dfnMales = dF[dF["applicant_sex_name"] != 2]
    
    # Estimation male / female
    dfTest["t-test Male&Others"] = stats.ttest_ind(dfnMales.iloc[:,0],dfMales.iloc[:,0])
    dfTest['f-test Male&Female&Others'] = stats.f_oneway(dfMales.iloc[:,0],dfFemales.iloc[:,0],dfO.iloc[:,0])
    
    
    # Initialisation etnicity
    dfWhite = dF[(dF["applicant_race_name_1"] == 6)]
    dfAsian = dF[(dF["applicant_race_name_1"] == 1)]
    dfBlack = dF[(dF["applicant_race_name_1"] == 2)]
    dfElse =  dF[(dF["applicant_race_name_1"] != 6) & (dF["applicant_race_name_1"] != 1) & (dF["applicant_race_name_1"] !=2)]
    dfnWhite = dF[(dF["applicant_race_name_1"] != 6)]
  
    
    dfTest['t-test White&Others'] = stats.ttest_ind(dfWhite.iloc[:,0],dfnWhite.iloc[:,0])
    dfTest['f-test White&Asian&Black&Else'] = stats.f_oneway(dfWhite.iloc[:,0],dfAsian.iloc[:,0],dfBlack.iloc[:,0],dfElse.iloc[:,0])
    dfTest['f-test White&Asian&Black'] = stats.f_oneway(dfWhite.iloc[:,0],dfAsian.iloc[:,0],dfBlack.iloc[:,0])
    
    col=["t-test Male&Others",'f-test Male&Female&Others','t-test White&Others','f-test White&Asian&Black&Else','f-test White&Asian&Black']
    
    return dfTest
    
###########################################################
### dY= emptyfunc(vX)
def Estfun_ML(sFun_ML,dfX, iDepth, iFeatures, iRandom, protected_attribute_name, privileged_num, sFairnessMetric ,IncludeProtected = True):
    """
    Purpose:
        Performs a Random Forest machine learning calculation.
        sFairnessMetric:
            [0] DisparateImpactRemover
            [1] Reweighting 
            [2] Massaging
            [3] Sampling
            [4] EqualizedOds
            [5] RejectOptionClassification

    Inputs:
        fun_ML                          Function, to call the desired maximum likelihood function
        dfX                             Dataframe, containing all observations and attributes
        iDepth                          Integer, level of tree depth
        iFeatures                       Integer, max number of features
        iRandom                         Integer, random seed
        protected_attribute_name        String, indicating the categorial variable which is a protected attribute
        privileged_num                  Integer, indicating the categorical value for the priviliged group
        sFairnessMetric                 String, indicating which type of pre-processing technique to use on the training dataset

    Return value:
        
    """
    fun_ML = StringToFunc(sFun_ML)
    
    # Redefine the protected attribute only as binary values. With 1 == priviledged group
    vI = dfX[protected_attribute_name] == privileged_num
    dfTrans = dfX.copy(deep=True)
    dfTrans[protected_attribute_name] = vI.astype(int)
    # Return a vTest = None if there are no adjustments to the dataset
    vTest = None
    # Check for Preprocessing techniques
    vDiscPre = ['DisparateImpactRemover', 'Reweighting', 'Massaging', 'Sampling']
    for i in vDiscPre:
        if i == sFairnessMetric:
            print("\n Found a True value for the following preprocessing technique:",i, "\n Adjusting the training data set accordingly.")
            dfTrans, vTest = AdjustPreprocessing(dfTrans,protected_attribute_name, privileged_num, sFairnessMetric)

    # Split the original dataset into training and test data
    dfTarget = dfTrans["TARGET"]
    dfTrain =  dfTrans.copy(deep=True)
    dfTrain.drop(['TARGET'],axis=1,inplace = True)
    dfX_train, dfX_test, dfY_train, dfY_test = train_test_split(dfTrain, dfTarget,random_state=iRandom)
    
    # Check if protected attribute should be removed or not
    if IncludeProtected != True:
        dfProtected_train = dfX_train[protected_attribute_name]
        dfX_train.drop([protected_attribute_name],axis=1,inplace = True) 
        dfProtected_test = dfX_test[protected_attribute_name]
        dfX_test.drop([protected_attribute_name],axis=1,inplace = True) 
        
    #Machine learning Classifier
    fun_MLe = fun_ML(max_features=iFeatures,max_depth=iDepth)
    print("Estimating the machine learning model")
    fun_MLe.fit(dfX_train, dfY_train)
    target_predict = fun_MLe.predict(dfX_test)
    target_predict = pd.DataFrame(target_predict)
    target_predict_proba = fun_MLe.predict_proba(dfX_test)
    target_predict_proba = pd.DataFrame(target_predict_proba)
    

    # Make bar chart of model attributes
    barplot = PlotaBar(dfX_test,fun_MLe,sFun_ML,IncludeProtected,sFairnessMetric,protected_attribute_name)
    
    if IncludeProtected != True:
        dfX_test[protected_attribute_name] = dfProtected_test
    
    
    # Check for Postprocessing techniques
    vDiscPost = ['EqualizedOds','RejectOptionClassification']
    for i in vDiscPost:
        if i == sFairnessMetric:
            print("\n Found a True value for the following postprocessing technique:",i, "\n Adjusting the data set accordingly.")
            dfX_test, target_predict = AdjustPostprocessing(dfX_test, dfY_test,target_predict,protected_attribute_name, privileged_num, sFairnessMetric)
    
    # Produce nice overview of performance and heatmap
    score = accuracy_score(dfY_test, target_predict)
    report = classification_report(dfY_test, target_predict)
    classifier_df = confusion_matrix(dfY_test, target_predict)
    heatmap = sns.heatmap(classifier_df, annot=True,fmt="d")

    # Make plot of distribution of outcomes depending on sensitive attribute 1 or 0
    distriplot = PlotaDistribution(dfX_test,target_predict_proba.iloc[:,-1],sFun_ML,IncludeProtected,sFairnessMetric,protected_attribute_name)
    
    print("Computing metrics based on the outcome of the Random Forest model")
    dfM = Metrics(dfX_test,protected_attribute_name, privileged_num, target_predict)
    
    #Extend dfM with additional comparison metrics (that also involve the predictions of the ML model)
    ComparisonMetric(dfM, dfX_test, dfY_test, target_predict, protected_attribute_name, privileged_num)
    

    return (fun_MLe, target_predict, score, report, classifier_df, vTest, dfM)

###########################################################
### dY= emptyfunc(vX)
def StringToFunc(sFun_ML):
    """
    Purpose:
        Transforms provided string into appropriate function name for machine learning algorithm
        
    Inputs:
        sFun_ML         String, provides name for the function
        
    Return value:
        fun_ML          function, after transformation from string to function
        
    """
    if sFun_ML == 'Random Forest':
        fun_ML = RandomForestClassifier
        print(sFun_ML, 'algorithm succesfully recognized.')
    elif sFun_ML == 'Decision Tree':
        fun_ML = DecisionTreeClassifier
        print(sFun_ML, 'algorithm succesfully recognized.')
    elif sFun_ML == 'Gradient Boosting':
        fun_ML = GradientBoostingClassifier
        print(sFun_ML, 'algorithm succesfully recognized.')
    else:
        fun_ML = sFun_ML
        print(sFun_ML, 'not succesfully recognized. \n', 'Program can still proceed if provided function is sk-learning compatible.' )
    
    return fun_ML

###########################################################
### dY= emptyfunc(vX)
def AdjustPreprocessing(dfX, protected_attribute_name, privileged_num, sFairnessMetric):
    """
    Purpose:
        Function to prepare the preprocessing techniques on the dataset.
        sFairnessMetric:
            [0] DisparateImpactRemover
            [1] Reweighting 
            [2] Massaging
            [3] Sampling

    Inputs:
        dfX                             Dataframe, containing all observations and attributes
        protected_attribute_name        String, indicating the categorial variable which is a protected attribute
        privileged_num                  Integer, indicating the categorical value for the priviliged group
        sFairnessMetric                 String, indicating which type of pre-processing technique to use on the training dataset

    Return value:
        
    """
    # Redefine the protected attribute only as binary values. With 1 == priviledged group
    BinLabel = TransformBinLabel(dfX,protected_attribute_name, privileged_num)

    # Create test vector to store some metrics for test data set
    vTest = np.zeros(3)
    
    # Disparate Impact Remover
    if sFairnessMetric == 'DisparateImpactRemover':
        DI = preprocessing.DisparateImpactRemover(repair_level=1,sensitive_attribute=protected_attribute_name).fit_transform(BinLabel)
        dfAdjusted = DI.convert_to_dataframe()[0]
        vTest[0] = BinaryLabelDatasetMetric(DI).base_rate()
        vTest[1] = BinaryLabelDatasetMetric(DI, privileged_groups=[{protected_attribute_name: 1}], unprivileged_groups=[{protected_attribute_name: 0}]).disparate_impact()
        vTest[2] = BinaryLabelDatasetMetric(DI, privileged_groups=[{protected_attribute_name: 1}], unprivileged_groups=[{protected_attribute_name: 0}]).mean_difference()
        print("Base rate training set: ", vTest[0])
        print("Disparate Impact training set:",vTest[1])
        print("Mean-difference training set:", vTest[2])
    # Reweighing
    if sFairnessMetric == 'Reweighting':
        RW = preprocessing.Reweighing(privileged_groups=[{protected_attribute_name: 1}], unprivileged_groups=[{protected_attribute_name: 0}]).fit_transform(BinLabel)
        dfAdjusted = RW.convert_to_dataframe()[0]
        vTest[0] = BinaryLabelDatasetMetric(RW).base_rate()
        vTest[1] = BinaryLabelDatasetMetric(RW, privileged_groups=[{protected_attribute_name: 1}], unprivileged_groups=[{protected_attribute_name: 0}]).disparate_impact()
        vTest[2] = BinaryLabelDatasetMetric(RW, privileged_groups=[{protected_attribute_name: 1}], unprivileged_groups=[{protected_attribute_name: 0}]).mean_difference()
        print("Base rate training set: ", vTest[0])
        print("Disparate Impact training set:",vTest[1])
        print("Mean-difference training set:", vTest[2])
    # Massaging
    
    # Sampling
    
    # Remove target column from training dataset
    # dfAdjusted.drop(["TARGET"],axis=1,inplace = True) 
    
    
    return dfAdjusted, vTest

###########################################################
### dY= emptyfunc(vX)
def AdjustPostprocessing(dfX, dfY, target,protected_attribute_name, privileged_num, sFairnessMetric):
    """
    Purpose:
        Function to prepare the postprocessing techniques on the dataset.
        sFairnessMetric:
            [0] EqualizedOds
            [1] RejectOptionClassification

    Inputs:
        dfX                             Dataframe, containing all observations and attributes
        iDepth                          Integer, level of tree depth
        iFeatures                       Integer, max number of features
        iRandom                         Integer, random seed
        protected_attribute_name        String, indicating the categorial variable which is a protected attribute
        privileged_num                  Integer, indicating the categorical value for the priviliged group
        sFairnessMetric                 String, indicating which type of pre-processing technique to use on the training dataset

    Return:
        
    """
    # Obtain BinLabel data sets to perform Postprocessing
    BinLabel_true = TransformBinLabel(dfX,protected_attribute_name, privileged_num, dfY)
    BinLabel_predicted = TransformBinLabel(dfX,protected_attribute_name, privileged_num, target)

    
    # EqualizedOdds
    if sFairnessMetric == 'EqualizedOds':
        EO = postprocessing.EqOddsPostprocessing(privileged_groups=[{protected_attribute_name: 1}], 
                                                unprivileged_groups=[{protected_attribute_name: 0}]).fit_predict(BinLabel_true,BinLabel_predicted)
        dfAdjusted = EO.convert_to_dataframe()[0]
    
    # Reject Option Classification
    if sFairnessMetric == 'RejectOptionClassification':
        ROC = postprocessing.RejectOptionClassification(privileged_groups=[{protected_attribute_name: 1}], 
                                                        unprivileged_groups=[{protected_attribute_name: 0}],
                                                        metric_name="Equal opportunity difference").fit_predict(BinLabel_true,BinLabel_predicted)
        dfAdjusted = ROC.convert_to_dataframe()[0]
        
        
    # Seperate the X and Y variables
    dftarget = dfAdjusted['TARGET']
    dfAdjusted.drop(['TARGET'],axis=1,inplace = True)

        
    return dfAdjusted, dftarget

###########################################################
### dY= emptyfunc(vX)
def Metrics(df,protected_attribute_name, privileged_num, *args):
    """
    Purpose:
        Used to compute the Metrics of the given dataset.
        Protected attributes must be given as categorial variables, with privileged_num as the priviliged group
        Dataframe should contain a column TARGET, or the TARGET column should be provided in the target input
    
    Inputs:
        df                              Dataframe, containing the dataset for which fairness metrics are computed
        protected_attribute_name        String, indicating the categorial variable which is a protected attribute
        privileged_num                  Integer, indicating the categorical value for the priviliged group
        target                          Dataframe if given it should hold binary values representative of the TARGET column
    
    Return value:
        dfMetrics                       Dataframe, holding the computed metrics
    """
    # Transform into BinLabel data set
    BinLabel = TransformBinLabel(df,protected_attribute_name, privileged_num, *args)
    
    # Compute metrics for dataset
    vMetrics = np.zeros(8)
    vMetrics[0] = BinaryLabelDatasetMetric(BinLabel).base_rate()
    # vMetrics[1] = BinaryLabelDatasetMetric(BinLabel).consistency()   
    vMetrics[2] = BinaryLabelDatasetMetric(BinLabel, privileged_groups=[{protected_attribute_name: 1}], unprivileged_groups=[{protected_attribute_name: 0}]).disparate_impact()
    vMetrics[3] = BinaryLabelDatasetMetric(BinLabel, privileged_groups=[{protected_attribute_name: 1}], unprivileged_groups=[{protected_attribute_name: 0}]).mean_difference()
    vMetrics[4] = BinaryLabelDatasetMetric(BinLabel).num_instances()
    vMetrics[5] = BinaryLabelDatasetMetric(BinLabel).num_negatives()
    vMetrics[6] = BinaryLabelDatasetMetric(BinLabel).num_positives()
    vMetrics[7] = BinaryLabelDatasetMetric(BinLabel).smoothed_empirical_differential_fairness()
    
    # Import into pd dataframe
    col = ["Base_rate","Consistency","Disparate_impact","Mean_difference","Num_instances","Num_negatives","Num_positives","Smoothed_empirical_differential_fairness"]
    dfMetrics = pd.DataFrame(vMetrics.reshape(1,-1), columns=col, index=None)
    return dfMetrics

###########################################################
### dY= emptyfunc(vX)
def TransformBinLabel(df,protected_attribute_name, privileged_num, target=[0]):
    """
    Purpose:
        Used to transform a PD dataframe into the binaryclasslabel needed for fairness computations.
        Also automatically appends new target column if one is given.
    
    Inputs:
        df                              Dataframe, containing the dataset for which fairness metrics are computed
        protected_attribute_name        String, indicating the categorial variable which is a protected attribute
        privileged_num                  Integer, indicating the categorical value for the priviliged group
        target                          Dataframe if given it should hold binary values representative of the TARGET column
    
    Return:
        BinLabel                        Dataset used for computing metrics
    """
    # Check if additional target column has to be added
    if len(target) != 1:
        # print("----------------------------------")
        # print("Found additional TARGET column. Will replace original TARGET column if one was provided in the original dataset.")
        # print("----------------------------------")
        print('.')
        df["TARGET"] = target.values
    # Import into BinaryLabelDataset
    BinLabel = BinaryLabelDataset(df=df, label_names=["TARGET"],
                                  protected_attribute_names=[protected_attribute_name])
    BinLabel.validate_dataset()
    
    return BinLabel


###########################################################
### dY= emptyfunc(vX)
def ComparisonMetric(dfM,dfX,dfY,target,protected_attribute_name, privileged_num):
    """
    Purpose: 
        Calculating different mesasures of the outcomes of a machine learning model 
        for a given protected attribute and priviliged group
   
    
    Inputs:
        dfM
        dfX                             Dataframe, containing the dependent variables (the attributes)
                                        where subscribt train and test indicate if the dataframe is used for training the ML model or testing
        dfY                             Dataframe, containing the independent variable (the outcome)
        targetpredict                   Vector, containing the predicted outcomes by the ML model for the testdata
        protected_attribute_name        String, indicating the categorial variable which is a protected attribute
        privileged_num                  Integer, indicating the categorical value for the priviliged group
                
   Output:
       dfM
    """
    
    BinLabel_true = TransformBinLabel(dfX,protected_attribute_name, privileged_num, dfY)
    BinLabel_predicted = TransformBinLabel(dfX,protected_attribute_name, privileged_num, target)
    
    dfM['AverageOddsDifference'] = ClassificationMetric(BinLabel_true,BinLabel_predicted,unprivileged_groups=[{protected_attribute_name: 0}],privileged_groups=[{protected_attribute_name: 1}]).average_odds_difference()
    dfM['AveragAbsOddsDifference']= ClassificationMetric(BinLabel_true,BinLabel_predicted,unprivileged_groups=[{protected_attribute_name: 0}],privileged_groups=[{protected_attribute_name: 1}]).average_abs_odds_difference()
    dfM['EqualOpportunityDifference']= ClassificationMetric(BinLabel_true,BinLabel_predicted,unprivileged_groups=[{protected_attribute_name: 0}],privileged_groups=[{protected_attribute_name: 1}]).equal_opportunity_difference()
    dfM['Errorrate'] =  ClassificationMetric(BinLabel_true,BinLabel_predicted,unprivileged_groups=[{protected_attribute_name: 0}],privileged_groups=[{protected_attribute_name: 1}]).error_rate()
    dfM['ErrorrateRatio'] =  ClassificationMetric(BinLabel_true,BinLabel_predicted,unprivileged_groups=[{protected_attribute_name: 0}],privileged_groups=[{protected_attribute_name: 1}]).error_rate_ratio()
    dfM['FalseDiscoveryRate'] = ClassificationMetric(BinLabel_true,BinLabel_predicted,unprivileged_groups=[{protected_attribute_name: 0}],privileged_groups=[{protected_attribute_name: 1}]).false_discovery_rate()
    dfM['FalseDiscoveryRatio'] = ClassificationMetric(BinLabel_true,BinLabel_predicted,unprivileged_groups=[{protected_attribute_name: 0}],privileged_groups=[{protected_attribute_name: 1}]).false_discovery_rate_ratio()
    dfM['FalseOmissionRate'] = ClassificationMetric(BinLabel_true,BinLabel_predicted,unprivileged_groups=[{protected_attribute_name: 0}],privileged_groups=[{protected_attribute_name: 1}]).false_omission_rate()
    dfM['FalseOmissionRatio'] = ClassificationMetric(BinLabel_true,BinLabel_predicted,unprivileged_groups=[{protected_attribute_name: 0}],privileged_groups=[{protected_attribute_name: 1}]).false_omission_rate_ratio()
    
    
###########################################################
### dY= emptyfunc(vX)
def MetricsDataset(df,protected_attribute_name, privileged_num):
    """
    Purpose:
        Used to compute the Metrics of the given dataset.
        Protected attributes must be given as categorial variables, with privileged_num as the priviliged group
        Dataframe should contain a column TARGET, or the TARGET column should be provided in the target input.
        Preps dataset to be used in the original Metrics function
    
    Inputs:
        df                              Dataframe, containing the dataset for which fairness metrics are computed
        protected_attribute_name        String, indicating the categorial variable which is a protected attribute
        privileged_num                  Integer, indicating the categorical value for the priviliged group
        target                          Dataframe if given it should hold binary values representative of the TARGET column
    
    Return value:
        dfMetrics                       Dataframe, holding the computed metrics
    """
    # Redefine the protected attribute only as binary values. With 1 == priviledged group
    vI = df[protected_attribute_name] == privileged_num
    dfTrans = df.copy(deep=True)
    dfTrans[protected_attribute_name] = vI.astype(int)
    
    # Compute metrics
    dfMetrics = Metrics(dfTrans,protected_attribute_name, privileged_num)
    
    return dfMetrics

###########################################################
### dY= emptyfunc(vX)
def PlotaBar(df,model_output,sName,Includeprotected,sFairnessMetric,sType):
    """
    Purpose:
        To plot a bar chart of the given machinelearning model and save it in WD.
    
    Inputs:
        df                              Dataframe, containing the dataset for which fairness metrics are computed
        model_output                    Sklearn ensamble, containing the classes of the machine learning model
        sName                           String, indicating the name of the machine learning model applied
        Includeprotected                Bool, indicating if the sensitive attribute was used
        sFairnessMetric                 String, indicating what fairness metric is used
    """
    if sFairnessMetric == None:
        sFairnessMetric = "None"

    col = df.columns

    features = model_output.feature_importances_

    index = np.arange(features.shape[0])
    plt.bar(col,features,0.5)
    plt.xticks(rotation=85)
    plt.title(sName)
    plt.savefig(sName+sType+"_"+str(Includeprotected)+"_"+sFairnessMetric)
    plt.show()

###########################################################
### dY= emptyfunc(vX)
def PlotaDistribution(dfX,dfY,sName,Includeprotected,sFairnessMetric,sType):
    """
    Purpose:
        To plot a histogram of the given dataframe and save it into WD.
    
    Inputs:
        df                              Dataframe, containing the dataset for which fairness metrics are computed
        sName                           String, indicating the name of the machine learning model applied
        Includeprotected                Bool, indicating if the sensitive attribute was used
        sFairnessMetric                 String, indicating what fairness metric is used
    """
    if sFairnessMetric == None:
        sFairnessMetric = "None"
    
    dfX['proba'] = dfY.values
        
    dfDeprived= dfX[dfX[sType]==0]
    dfFavoured= dfX[dfX[sType]==1]
    
    fig, ax = plt.subplots(1,2, figsize=(15,8))
    
    ax[0].hist(dfDeprived['proba'],bins=100,density=True)
    # ax[0].savefig('Hist'+sName+sType+"_"+str(Includeprotected)+"_"+sFairnessMetric+"_Deprived")
    ax[0].set_ylabel('Density')
    ax[0].set_xlabel('Probability of loan approval')
    ax[0].set_title('Deprived group')

    
    ax[1].hist(dfFavoured['proba'],bins=100,density=True)
    # ax[1].savefig('Hist'+sName+sType+"_"+str(Includeprotected)+"_"+sFairnessMetric+"_Favoured")
    ax[1].set_ylabel('Density')
    ax[1].set_xlabel('Probability of loan approval')
    ax[1].set_title('Favoured group')
    
    
    fig.tight_layout(pad=1.5)
    fig.savefig('Hist_'+sName+sType+""+str(Includeprotected)+""+sFairnessMetric+"_Favoured")
    fig.show()
    
  