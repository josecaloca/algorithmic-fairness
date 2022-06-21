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
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms import preprocessing
from aif360.algorithms import postprocessing
from xgboost import XGBClassifier

###########################################################
### dY= emptyfunc(vX)
def cramers_V(var1, var2):
    crosstab = np.array(
        pd.crosstab(var1, var2, rownames=None, colnames=None)
    )  # Cross table building
    stat = stats.chi2_contingency(crosstab)[
        0
    ]  # Keeping of the test statistic of the Chi2 test
    obs = np.sum(crosstab)  # Number of observations
    mini = (
        min(crosstab.shape) - 1
    )  # Take the minimum value between the columns and the rows of the cross table
    return stat / (obs * mini)


###########################################################
### dY= emptyfunc(vX)
def count_outliers(data=pd.DataFrame(), variable=str()):
    mean_income = data[variable].mean()
    sd_income = data[variable].std()
    cut_off = mean_income + 3 * sd_income
    num_outliers = sum(df[variable] > cut_off)
    return num_outliers


###########################################################
### dY= emptyfunc(vX)
def iv_woe(data, target, bins=10, show_woe=False):

    # Empty Dataframe
    newDF, woeDF = pd.DataFrame(), pd.DataFrame()

    # Extract Column Names
    cols = data.columns

    # Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in "bifc") and (len(np.unique(data[ivars])) > 10):
            binned_x = pd.qcut(data[ivars], bins, duplicates="drop")
            d0 = pd.DataFrame({"x": binned_x, "y": data[target]})
        else:
            d0 = pd.DataFrame({"x": data[ivars], "y": data[target]})

        # Calculate the number of events in each group (bin)
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ["Cutoff", "N", "Events"]

        # Calculate % of events in each group.
        d["% of Events"] = np.maximum(d["Events"], 0.5) / d["Events"].sum()

        # Calculate the non events in each group.
        d["Non-Events"] = d["N"] - d["Events"]
        # Calculate % of non events in each group.
        d["% of Non-Events"] = np.maximum(d["Non-Events"], 0.5) / d["Non-Events"].sum()

        # Calculate WOE by taking natural log of division of % of non-events and % of events
        d["WoE"] = np.log(d["% of Events"] / d["% of Non-Events"])
        d["IV"] = d["WoE"] * (d["% of Events"] - d["% of Non-Events"])
        d.insert(loc=0, column="Variable", value=ivars)
        print("Information value of " + ivars + " is " + str(round(d["IV"].sum(), 6)))
        temp = pd.DataFrame(
            {"Variable": [ivars], "IV": [d["IV"].sum()]}, columns=["Variable", "IV"]
        )
        newDF = pd.concat([newDF, temp], axis=0)
        woeDF = pd.concat([woeDF, d], axis=0)

        # Show WOE Table
        if show_woe == True:
            print(d)
    return newDF, woeDF


###########################################################
### dY= emptyfunc(vX)
def central_tendency_tests(dfZ):
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

    dF = dfZ[["TARGET", "applicant_sex", "applicant_race_1", "applicant_ethnicity"]]

    # Initialisation sex
    dfFemales = dF[dF["applicant_sex"] == 2]
    dfMales = dF[dF["applicant_sex"] == 1]
    dfO = dF[(dF["applicant_sex"] != 1) & (dF["applicant_sex"] != 2)]
    dfnMales = dF[dF["applicant_sex"] != 1]

    dfTest["t-test Male&Female"] = stats.ttest_ind(
        dfFemales.iloc[:, 0], dfMales.iloc[:, 0]
    )
    dfTest["f-test Male&Female&Others"] = stats.f_oneway(
        dfMales.iloc[:, 0], dfFemales.iloc[:, 0], dfO.iloc[:, 0]
    )

    # Initialisation race
    dfWhite = dF[(dF["applicant_race_1"] == 5)]
    dfAsian = dF[(dF["applicant_race_1"] == 2)]
    dfBlack = dF[(dF["applicant_race_1"] == 3)]
    dfElse = dF[
        (dF["applicant_race_1"] != 5)
        & (dF["applicant_race_1"] != 2)
        & (dF["applicant_race_1"] != 3)
    ]
    dfnWhite = dF[(dF["applicant_race_1"] != 5)]

    dfTest["t-test White&Others"] = stats.ttest_ind(
        dfWhite.iloc[:, 0], dfnWhite.iloc[:, 0]
    )
    dfTest["t-test White&Blacks"] = stats.ttest_ind(
        dfWhite.iloc[:, 0], dfBlack.iloc[:, 0]
    )
    dfTest["f-test White&Asian&Black&Else"] = stats.f_oneway(
        dfWhite.iloc[:, 0], dfAsian.iloc[:, 0], dfBlack.iloc[:, 0], dfElse.iloc[:, 0]
    )
    dfTest["f-test White&Asian&Black"] = stats.f_oneway(
        dfWhite.iloc[:, 0], dfAsian.iloc[:, 0], dfBlack.iloc[:, 0]
    )

    # Initialisation ethnicity
    dfLatino = dF[dF["applicant_ethnicity"] == 1]
    dfOethnicity = dF[dF["applicant_ethnicity"] == 2]
    dfnLatino = dF[dF["applicant_sex"] != 1]

    dfTest["t-test Latino&Otherethnicity"] = stats.ttest_ind(
        dfLatino.iloc[:, 0], dfOethnicity.iloc[:, 0]
    )
    dfTest["f-test Latino&Otherethnicity"] = stats.f_oneway(
        dfLatino.iloc[:, 0], dfOethnicity.iloc[:, 0], dfnLatino.iloc[:, 0]
    )

    dfTest = dfTest.T
    dfTest.rename(columns={0: "statistic", 1: "p-value"}, inplace=True)

    return dfTest


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
    if sFun_ML == "Random Forest":
        fun_ML = RandomForestClassifier
        print(sFun_ML, "algorithm succesfully recognized.")
    elif sFun_ML == "Logistic Regression":
        fun_ML = LogisticRegression
        print(sFun_ML, "algorithm succesfully recognized.")
    elif sFun_ML == "XGBoost":
        fun_ML = XGBClassifier
        print(sFun_ML, "algorithm succesfully recognized.")
    else:
        fun_ML = sFun_ML
        print(
            sFun_ML,
            "not succesfully recognized. \n",
            "Program can still proceed if provided function is sk-learning compatible.",
        )

    return fun_ML


###########################################################
### dY= emptyfunc(vX)
def AdjustPreprocessing(
    dfX, protected_attribute_name, privileged_num, unprivileged_num, sFairnessMetric
):
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
    BinLabel = TransformBinLabel(dfX, protected_attribute_name, privileged_num)

    # Create test vector to store some metrics for test data set
    vTest = np.zeros(3)

    # Disparate Impact Remover
    if sFairnessMetric == "DisparateImpactRemover":
        DI = preprocessing.DisparateImpactRemover(
            repair_level=1, sensitive_attribute=protected_attribute_name
        ).fit_transform(BinLabel)
        dfAdjusted = DI.convert_to_dataframe()[0]
        vTest[0] = BinaryLabelDatasetMetric(DI).base_rate()
        vTest[1] = BinaryLabelDatasetMetric(
            DI,
            privileged_groups=[{protected_attribute_name: privileged_num}],
            unprivileged_groups=[{protected_attribute_name: unprivileged_num}],
        ).disparate_impact()
        vTest[2] = BinaryLabelDatasetMetric(
            DI,
            privileged_groups=[{protected_attribute_name: privileged_num}],
            unprivileged_groups=[{protected_attribute_name: unprivileged_num}],
        ).mean_difference()
        print("Base rate training set: ", vTest[0])
        print("Disparate Impact training set:", vTest[1])
        print("Mean-difference training set:", vTest[2])
    # Reweighing
    if sFairnessMetric == "Reweighting":
        RW = preprocessing.Reweighing(
            privileged_groups=[{protected_attribute_name: privileged_num}],
            unprivileged_groups=[{protected_attribute_name: unprivileged_num}],
        ).fit_transform(BinLabel)
        dfAdjusted = RW.convert_to_dataframe()[0]
        vTest[0] = BinaryLabelDatasetMetric(RW).base_rate()
        vTest[1] = BinaryLabelDatasetMetric(
            RW,
            privileged_groups=[{protected_attribute_name: privileged_num}],
            unprivileged_groups=[{protected_attribute_name: unprivileged_num}],
        ).disparate_impact()
        vTest[2] = BinaryLabelDatasetMetric(
            RW,
            privileged_groups=[{protected_attribute_name: privileged_num}],
            unprivileged_groups=[{protected_attribute_name: unprivileged_num}],
        ).mean_difference()
        print("Base rate training set: ", vTest[0])
        print("Disparate Impact training set:", vTest[1])
        print("Mean-difference training set:", vTest[2])

    return dfAdjusted, vTest


###########################################################
### dY= emptyfunc(vX)
def AdjustPostprocessing(
    dfX,
    dfY,
    target,
    protected_attribute_name,
    privileged_num,
    unprivileged_num,
    sFairnessMetric,
):
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
    BinLabel_true = TransformBinLabel(
        dfX, protected_attribute_name, privileged_num, dfY
    )
    BinLabel_predicted = TransformBinLabel(
        dfX, protected_attribute_name, privileged_num, target
    )

    # EqualizedOdds
    if sFairnessMetric == "EqualizedOds":
        EO = postprocessing.EqOddsPostprocessing(
            privileged_groups=[{protected_attribute_name: privileged_num}],
            unprivileged_groups=[{protected_attribute_name: unprivileged_num}],
        ).fit_predict(BinLabel_true, BinLabel_predicted)
        dfAdjusted = EO.convert_to_dataframe()[0]

    # Reject Option Classification
    if sFairnessMetric == "RejectOptionClassification":
        ROC = postprocessing.RejectOptionClassification(
            privileged_groups=[{protected_attribute_name: privileged_num}],
            unprivileged_groups=[{protected_attribute_name: unprivileged_num}],
            metric_name="Statistical parity difference",
            metric_ub=0.05,
            metric_lb=-0.05,
            low_class_thresh=0.01,
            high_class_thresh=0.99,
            num_class_thresh=100,
            num_ROC_margin=50,
        ).fit_predict(BinLabel_true, BinLabel_predicted)
        dfAdjusted = ROC.convert_to_dataframe()[0]

    # Seperate the X and Y variables
    dftarget = dfAdjusted["TARGET"]
    dfAdjusted.drop(["TARGET"], axis=1, inplace=True)

    return dfAdjusted, dftarget


###########################################################
### dY= emptyfunc(vX)
def Metrics(df, protected_attribute_name, privileged_num, unprivileged_num, *args):
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
    BinLabel = TransformBinLabel(df, protected_attribute_name, privileged_num, *args)

    # Compute metrics for dataset
    vMetrics = np.zeros(8)
    vMetrics[0] = BinaryLabelDatasetMetric(BinLabel).base_rate()
    # vMetrics[1] = BinaryLabelDatasetMetric(BinLabel).consistency()
    vMetrics[2] = BinaryLabelDatasetMetric(
        BinLabel,
        privileged_groups=[{protected_attribute_name: privileged_num}],
        unprivileged_groups=[{protected_attribute_name: unprivileged_num}],
    ).disparate_impact()
    vMetrics[3] = BinaryLabelDatasetMetric(
        BinLabel,
        privileged_groups=[{protected_attribute_name: privileged_num}],
        unprivileged_groups=[{protected_attribute_name: unprivileged_num}],
    ).mean_difference()
    vMetrics[4] = BinaryLabelDatasetMetric(BinLabel).num_instances()
    vMetrics[5] = BinaryLabelDatasetMetric(BinLabel).num_negatives()
    vMetrics[6] = BinaryLabelDatasetMetric(BinLabel).num_positives()
    vMetrics[7] = BinaryLabelDatasetMetric(
        BinLabel
    ).smoothed_empirical_differential_fairness()

    # Import into pd dataframe
    col = [
        "Base_rate",
        "Consistency",
        "Disparate_impact",
        "Mean_difference",
        "Num_instances",
        "Num_negatives",
        "Num_positives",
        "Smoothed_empirical_differential_fairness",
    ]
    dfMetrics = pd.DataFrame(vMetrics.reshape(1, -1), columns=col, index=None)
    return dfMetrics


###########################################################
### dY= emptyfunc(vX)
def TransformBinLabel(df, protected_attribute_name, privileged_num, target=[0]):
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
        print(".")
        df["TARGET"] = target.values
    # Import into BinaryLabelDataset
    BinLabel = BinaryLabelDataset(
        df=df,
        label_names=["TARGET"],
        protected_attribute_names=[protected_attribute_name],
    )
    BinLabel.validate_dataset()

    return BinLabel


###########################################################
### dY= emptyfunc(vX)
def ComparisonMetric(
    dfM, dfX, dfY, target, protected_attribute_name, privileged_num, unprivileged_num
):
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

    BinLabel_true = TransformBinLabel(
        dfX, protected_attribute_name, privileged_num, dfY
    )
    BinLabel_predicted = TransformBinLabel(
        dfX, protected_attribute_name, privileged_num, target
    )

    dfM["AverageOddsDifference"] = ClassificationMetric(
        BinLabel_true,
        BinLabel_predicted,
        unprivileged_groups=[{protected_attribute_name: unprivileged_num}],
        privileged_groups=[{protected_attribute_name: privileged_num}],
    ).average_odds_difference()
    dfM["AveragAbsOddsDifference"] = ClassificationMetric(
        BinLabel_true,
        BinLabel_predicted,
        unprivileged_groups=[{protected_attribute_name: unprivileged_num}],
        privileged_groups=[{protected_attribute_name: privileged_num}],
    ).average_abs_odds_difference()
    dfM["EqualOpportunityDifference"] = ClassificationMetric(
        BinLabel_true,
        BinLabel_predicted,
        unprivileged_groups=[{protected_attribute_name: unprivileged_num}],
        privileged_groups=[{protected_attribute_name: privileged_num}],
    ).equal_opportunity_difference()
    dfM["Errorrate"] = ClassificationMetric(
        BinLabel_true,
        BinLabel_predicted,
        unprivileged_groups=[{protected_attribute_name: unprivileged_num}],
        privileged_groups=[{protected_attribute_name: privileged_num}],
    ).error_rate()
    dfM["ErrorrateRatio"] = ClassificationMetric(
        BinLabel_true,
        BinLabel_predicted,
        unprivileged_groups=[{protected_attribute_name: unprivileged_num}],
        privileged_groups=[{protected_attribute_name: privileged_num}],
    ).error_rate_ratio()
    dfM["FalseDiscoveryRate"] = ClassificationMetric(
        BinLabel_true,
        BinLabel_predicted,
        unprivileged_groups=[{protected_attribute_name: unprivileged_num}],
        privileged_groups=[{protected_attribute_name: privileged_num}],
    ).false_discovery_rate()
    dfM["FalseDiscoveryRatio"] = ClassificationMetric(
        BinLabel_true,
        BinLabel_predicted,
        unprivileged_groups=[{protected_attribute_name: unprivileged_num}],
        privileged_groups=[{protected_attribute_name: privileged_num}],
    ).false_discovery_rate_ratio()
    dfM["FalseOmissionRate"] = ClassificationMetric(
        BinLabel_true,
        BinLabel_predicted,
        unprivileged_groups=[{protected_attribute_name: unprivileged_num}],
        privileged_groups=[{protected_attribute_name: privileged_num}],
    ).false_omission_rate()
    dfM["FalseOmissionRatio"] = ClassificationMetric(
        BinLabel_true,
        BinLabel_predicted,
        unprivileged_groups=[{protected_attribute_name: unprivileged_num}],
        privileged_groups=[{protected_attribute_name: privileged_num}],
    ).false_omission_rate_ratio()


###########################################################
### dY= emptyfunc(vX)
def MetricsDataset(df, protected_attribute_name, privileged_num):
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
    dfMetrics = Metrics(dfTrans, protected_attribute_name, privileged_num)

    return dfMetrics


###########################################################
### dY= emptyfunc(vX)
def PlotaBar(df, model_output, sName, Includeprotected, sFairnessMetric, sType):
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
    d = {"variables": col, "values": features}
    var_importance = pd.DataFrame(d)
    var_importance = var_importance.sort_values(by=["values"])
    var_importance.set_index("variables").plot.bar()
    plt.show()


###########################################################
### dY= emptyfunc(vX)
def PlotaDistribution(
    dfX, dfY, sFairnessMetric, sType, actual_flag, unprivileged_num, privileged_num
):
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

    dfX["proba"] = dfY.values
    dfX["actual_value"] = actual_flag.values

    dfDeprived = dfX[dfX[sType] == unprivileged_num]
    dfFavoured = dfX[dfX[sType] == privileged_num]

    # get mean of predicted probabilities
    mean_deprived_pred = dfDeprived["proba"].mean()
    mean_favoured_pred = dfFavoured["proba"].mean()

    # get observed probability
    mean_deprived_actual = dfDeprived["actual_value"].mean()
    mean_favoured_actual = dfFavoured["actual_value"].mean()

    # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
    # sns.set(style="darkgrid")
    plt.subplots(figsize=(14, 7))
    # plotting both distibutions on the same figure
    fig = sns.kdeplot(dfDeprived["proba"], shade=True, color="r")
    fig = sns.kdeplot(dfFavoured["proba"], shade=True, color="b")

    plt.axvline(mean_deprived_actual, 0, 1, color="r", linestyle="dashed")
    plt.axvline(mean_favoured_actual, 0, 1, color="b", linestyle="dashed")

    plt.axvline(mean_deprived_pred, 0, 1, color="r")
    plt.axvline(mean_favoured_pred, 0, 1, color="b")

    # plt.ylim(0, 10)

    plt.xlabel("Probability")
    plt.legend(
        labels=[
            "Black applicants",
            "White applicants",
            "Observed Blacks",
            "Observed Whites",
            "Predicted Blacks",
            "Predicted Whites",
        ]
    )
    # plt.show()
    # fig.savefig("./Graphs/"+'Hist_'+sName+sType+""+str(Includeprotected)+""+sFairnessMetric+"_Favoured")


def plot_ROC(y_train_true, y_train_prob, y_test_true, y_test_prob):
    """
    Purpose:
        Plot the ROC curve for train labels and test labels. Use the best threshold found in train set to classify items in test set.

    Inputs:
        y_train_true                        Array, containing true binary labels (default / no default) from the training set
        y_train_prob                        Array, containing predicted probabilities from the training set
        y_test_true                         Array, containing true binary labels (default / no default) from the test set
        y_test_prob                         Array, containing predicted probabilities from the test set


    Return value:

    """

    fpr_train, tpr_train, thresholds_train = roc_curve(
        y_train_true, y_train_prob, pos_label=True
    )
    sum_sensitivity_specificity_train = tpr_train + (1 - fpr_train)
    best_threshold_id_train = np.argmax(sum_sensitivity_specificity_train)
    best_threshold = thresholds_train[best_threshold_id_train]
    best_fpr_train = fpr_train[best_threshold_id_train]
    best_tpr_train = tpr_train[best_threshold_id_train]
    y_train = y_train_prob > best_threshold

    cm_train = confusion_matrix(y_train_true, y_train)
    auc_train = roc_auc_score(y_train_true, y_train)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    curve1 = ax.plot(fpr_train, tpr_train)
    curve2 = ax.plot([0, 1], [0, 1], color="navy", linestyle="--")
    dot = ax.plot(best_fpr_train, best_tpr_train, marker="o", color="black")
    ax.text(
        best_fpr_train,
        best_tpr_train,
        s="(%.3f,%.3f)" % (best_fpr_train, best_tpr_train),
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve (Train), AUC = %.4f" % auc_train)

    fpr_test, tpr_test, thresholds_test = roc_curve(
        y_test_true, y_test_prob, pos_label=True
    )

    y_test = y_test_prob > best_threshold

    cm_test = confusion_matrix(y_test_true, y_test)
    auc_test = roc_auc_score(y_test_true, y_test)

    tpr_score = float(cm_test[1][1]) / (cm_test[1][1] + cm_test[1][0])
    fpr_score = float(cm_test[0][1]) / (cm_test[0][0] + cm_test[0][1])

    ax2 = fig.add_subplot(122)
    curve1 = ax2.plot(fpr_test, tpr_test)
    curve2 = ax2.plot([0, 1], [0, 1], color="navy", linestyle="--")
    dot = ax2.plot(fpr_score, tpr_score, marker="o", color="black")
    ax2.text(fpr_score, tpr_score, s="(%.3f,%.3f)" % (fpr_score, tpr_score))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve (Test), AUC = %.4f" % auc_test)
    # plt.savefig('ROC', dpi = 500)
    plt.show()

    return best_threshold, auc_train, auc_test


######################################################################

# function for calculating fairness metrics defined in page 56 of the paper "Veritas Document 3A FEAT Fairness Principles Assessment Methodology"
def evaluate_performance(
    results_df,
    target_variable,
    predicted_variable,
    protected_variable,
    privileged_num,
    unprivileged_num,
):
    BELGIANS_LABEL, FOREIGNERS_LABEL = (privileged_num, unprivileged_num)
    # create accurate variable
    results_df["accurate"] = (
        results_df[target_variable] == results_df[predicted_variable]
    )

    # create an empty dictionary to store the fairness metrics

    summaries = {}

    # overall accuracy of the model
    overall_accuracy = results_df.accurate.mean()
    summaries["accuracy_overall"] = overall_accuracy

    # overall f1_measure of the model
    overall_f_measure = f1_score(
        results_df[target_variable], results_df[predicted_variable]
    )
    summaries["f1_score_overall"] = overall_f_measure

    # accuracy (aka positive predictive rate)
    for nationality in [(BELGIANS_LABEL, "White"), (FOREIGNERS_LABEL, "Black")]:

        # filter dataset per nationality group
        rows = results_df[results_df[protected_variable] == nationality[0]]

        # get confusion matrix
        tn, fp, fn, tp = confusion_matrix(
            rows[target_variable], rows[predicted_variable],
        ).ravel()

        # accuracy
        accuracy_for_nationality = (tp + tn) / (tp + fp + tn + fn)
        # false negative rate
        false_negative_rate = fn / (tp + fn)
        # false omission rate
        false_omission_rate = fn / (tn + fn)
        # true positive rate aka RECALL
        true_positive_rate = tp / (tp + fn)
        # positive predictive value aka PRECISION
        positive_predictive_value = tp / (tp + fp)
        # f1 score
        f_measure = 2 * (
            (positive_predictive_value * true_positive_rate)
            / (positive_predictive_value + true_positive_rate)
        )
        # false positive rate
        false_positive_rate = fp / (tn + fp)
        # false discovery rate
        false_discovery_rate = fp / (fp + tp)
        # average odds
        average_odds = (tp / (tp + fn) + fp / (tn + fp)) / 2
        # prevelence rate
        prevelence_rate = (tp + fn) / (fp + tp + tn + fn)

        # add results to the dictionary
        summaries["f1_score_" + nationality[1]] = f_measure
        summaries["accuracy_" + nationality[1]] = accuracy_for_nationality
        summaries["false_negative_rate_" + nationality[1]] = false_negative_rate
        summaries["false_omission_rate_" + nationality[1]] = false_omission_rate
        summaries["true_positive_rate_" + nationality[1]] = true_positive_rate
        summaries[
            "positive_predictive_value_" + nationality[1]
        ] = positive_predictive_value
        summaries["false_positive_rate_" + nationality[1]] = false_positive_rate
        summaries["false_discovery_rate_" + nationality[1]] = false_discovery_rate
        summaries["average_odds_" + nationality[1]] = average_odds
        summaries["prevelence_rate_" + nationality[1]] = prevelence_rate

    return summaries


# function for plotting results for 2 groups stacked togethey


def plot_performance_per_group(
    accuracy_results,
    title,
    fignum=1,
    rotation="horizontal",
    labels=["Whites", "Blacks"],
):

    assert isinstance(accuracy_results, list), "Accuracy results must be a list"

    indices = [0]
    colors = ["red", "blue"]
    fig, ax = plt.subplots()

    for index in indices:
        ax.scatter(
            index,
            accuracy_results[0][index],
            c=colors[0],
            label=labels[0] if labels and index == 0 else None,
        )
        ax.scatter(
            index,
            accuracy_results[1][index],
            c=colors[1],
            label=labels[1] if labels and index == 0 else None,
        )

    if labels:
        ax.legend()

    # plt.xticks(indices, approaches, rotation=rotation)
    plt.title(title)

    # plt.show()


# function for Plotting results for 2 groups stacked together


def plot_comparisons_groups(
    approaches,
    accuracy_results,
    title,
    fignum=1,
    rotation="horizontal",
    labels=["Whites", "Blacks"],
):

    assert isinstance(accuracy_results, list), "Accuracy results must be a list"

    indices = list(range(len(approaches)))
    colors = ["red", "blue"]
    fig, ax = plt.subplots(figsize=(14, 7))

    for index in indices:
        ax.scatter(
            index,
            accuracy_results[0][index],
            c=colors[0],
            label=labels[0] if labels and index == 0 else None,
        )
        ax.scatter(
            index,
            accuracy_results[1][index],
            c=colors[1],
            label=labels[1] if labels and index == 0 else None,
        )

    if labels:
        ax.legend()

    plt.ylim(0, 1)
    plt.xticks(indices, approaches, rotation=rotation)
    plt.title(title)

    # plt.show()


def plot_model_nationality_metrics(
    _feature, _summaries, _modelNames, _title, rotation="vertical"
):
    gender_metrics = [
        [summary[_feature + "_White"] for summary in _summaries],
        [summary[_feature + "_Black"] for summary in _summaries],
    ]
    plot_comparisons_groups(_modelNames, gender_metrics, _title, rotation=rotation)


def model_summary(model_name, title, summary):
    summaries = []
    model_names = []

    for key in [
        "f1_score",
        "accuracy",
        "false_negative_rate",
        "false_omission_rate",
        "true_positive_rate",
        "positive_predictive_value",
        "false_positive_rate",
        "false_discovery_rate",
        "average_odds",
        "prevelence_rate",
    ]:
        new_summary = {
            "accuracy_White": summary[key + "_White"],
            "accuracy_Black": summary[key + "_Black"],
        }
        summaries.append(new_summary)
        model_names.append(key)
    plot_model_nationality_metrics("accuracy", summaries, model_names, model_name)


######################################################################


###########################################################
### dY= emptyfunc(vX)
def Estfun_ML(
    sFun_ML,
    dfX,
    iDepth,
    iRandom,
    protected_attribute_name,
    unprivileged_num,
    privileged_num,
    sFairnessMetric,
    IncludeProtected,
):
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
    ##################
    # Data Check
    ##################
    dfTrans = dfX.copy()
    vTest = None

    # create flag of num of variables for tree-based algos
    iFeatures = len(dfTrans.columns)

    if IncludeProtected != True:
        if sFairnessMetric == "Reweighting":
            iFeatures = iFeatures - 2
        else:
            iFeatures = iFeatures - 1

    ##################
    # Pre-processing
    ##################

    if sFairnessMetric in ["Reweighting", "DisparateImpactRemover"]:

        # convert pandas dataset into aif360 dataset
        BinLabel = BinaryLabelDataset(
            df=dfTrans,
            label_names=["TARGET"],
            protected_attribute_names=[protected_attribute_name],
            favorable_label=0,
            unfavorable_label=1,
        )

        if sFairnessMetric == "Reweighting":
            # train a processor to get weights
            RW = preprocessing.Reweighing(
                privileged_groups=[{protected_attribute_name: privileged_num}],
                unprivileged_groups=[{protected_attribute_name: unprivileged_num}],
            ).fit_transform(BinLabel)
            # extract weights
            weights = RW.instance_weights.ravel()
            # add weights to the dataset
            dfTrans["WEIGHTS"] = weights

        elif sFairnessMetric == "DisparateImpactRemover":
            # train a processor fair disparate impact and change dataset structure
            DI = preprocessing.DisparateImpactRemover(repair_level=1).fit_transform(
                BinLabel
            )
            # convert dataframe into pandas DF
            dfTrans = DI.convert_to_dataframe()[0]

    ##################
    # Model Training
    ##################

    # Split the original dataset into training and test data. Stratified split selected to ensure same proportion in training and test sets
    dfTarget = dfTrans["TARGET"]
    dfTrain = dfTrans.copy(deep=True)
    dfTrain.drop(["TARGET"], axis=1, inplace=True)

    # Split dataset into train and test
    dfX_train, dfX_test, dfY_train, dfY_test = train_test_split(
        dfTrain, dfTarget, test_size=0.3, stratify=dfTarget, random_state=iRandom
    )

    # Check if protected attribute should be removed or not
    if IncludeProtected != True:
        dfProtected_train = dfX_train[protected_attribute_name]
        dfX_train.drop([protected_attribute_name], axis=1, inplace=True)
        dfProtected_test = dfX_test[protected_attribute_name]
        dfX_test.drop([protected_attribute_name], axis=1, inplace=True)

    # Set the model type

    fun_ML = StringToFunc(sFun_ML)
    if sFun_ML == "Logistic Regression":
        # Logistic Regression
        fun_MLe = fun_ML(random_state=iRandom)
        print(f"Estimating the {sFun_ML} model")
    else:
        # Tree Based Machine learning Classifier
        fun_MLe = fun_ML(max_features=iFeatures, max_depth=iDepth, n_estimators=688)
        print(f"Estimating the {sFun_ML} model")

    # Train a model
    if sFairnessMetric == "Reweighting":
        weights_reg = dfX_train["WEIGHTS"]
        dfX_train.drop(["WEIGHTS"], axis=1, inplace=True)
        dfX_test.drop(["WEIGHTS"], axis=1, inplace=True)
        print("training a weighted model")
        fun_MLe.fit(dfX_train, dfY_train, sample_weight=weights_reg)
    else:
        print("training a ML model")
        fun_MLe.fit(dfX_train, dfY_train)

    ##################
    # Model Predictions: AUC + Feature Importance
    ##################

    y_train_prob = pd.DataFrame(fun_MLe.predict_proba(dfX_train)[:, 1])
    pred_continuous = pd.DataFrame(fun_MLe.predict_proba(dfX_test)[:, 1])
    cut_off_point, auc_train, auc_test = plot_ROC(
        y_train_true=dfY_train,
        y_train_prob=y_train_prob,
        y_test_true=dfY_test,
        y_test_prob=pred_continuous,
    )
    target_predict = (pred_continuous >= cut_off_point).astype("int")
    # Make bar chart of feature importance
    if sFun_ML in ["Random Forest", "Gradient Boosting"]:
        barplot = PlotaBar(
            dfX_test,
            fun_MLe,
            sFun_ML,
            IncludeProtected,
            sFairnessMetric,
            protected_attribute_name,
        )

    # Remove the index of NATIONALITY and insert in the Metrics function

    # dfX_test.reset_index(inplace=True)

    ##################
    # Post-processing
    ##################
    # In case of a fair model, we include back to the dataset the protected attributes to perform further calculations
    if IncludeProtected != True:
        dfX_train[protected_attribute_name] = dfProtected_train
        dfX_test[protected_attribute_name] = dfProtected_test

    # Check for Postprocessing techniques
    vDiscPost = ["EqualizedOds", "RejectOptionClassification"]
    for i in vDiscPost:
        if i == sFairnessMetric:
            print(
                "\n Found a True value for the following postprocessing technique:",
                i,
                "\n Adjusting the data set accordingly.",
            )
            dfX_test, target_predict = AdjustPostprocessing(
                dfX_test,
                dfY_test,
                target_predict,
                protected_attribute_name,
                privileged_num,
                unprivileged_num,
                sFairnessMetric,
            )

    # Make plot of distribution of outcomes depending on sensitive attribute
    distriplot = PlotaDistribution(
        dfX=dfX_test,
        dfY=pred_continuous,
        sFairnessMetric=sFairnessMetric,
        sType=protected_attribute_name,
        actual_flag=dfY_test,
        unprivileged_num=unprivileged_num,
        privileged_num=privileged_num,
    )
    plt.show()
    # Create confussion matrix
    score = accuracy_score(dfY_test, target_predict)
    report = classification_report(dfY_test, target_predict)
    classifier_df = confusion_matrix(dfY_test, target_predict)
    # Plot confussion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=classifier_df, display_labels=fun_MLe.classes_
    )
    disp.plot()

    ########################
    # the function above add 1 var called "prob" that is the predicted continuous prob of default
    # target_predict is the predicted label after pre or post processing to which we are measuring fairness
    dfX_test["TARGET"] = target_predict.values

    dfM = Metrics(dfX_test, protected_attribute_name, privileged_num, unprivileged_num)
    # Extend dfM with additional comparison metrics (that also involve the predictions of the ML model)
    ComparisonMetric(
        dfM,
        dfX_test,
        dfY_test,
        target_predict,
        protected_attribute_name,
        privileged_num,
        unprivileged_num,
    )

    dfM = dfM.T.rename_axis("Fairness Metrics").rename(columns={0: "Values for Race"})

    dfM.reset_index(inplace=True)

    # SECTION TO IMPROVE: I must change the original code and set different variable names.
    dfX_test.rename(columns={"TARGET": "PREDS"}, inplace=True)
    dfX_test["TARGET"] = dfY_test.values

    ### Model Performance Metrics
    original_approach = evaluate_performance(
        dfX_test,
        target_variable="TARGET",
        predicted_variable="PREDS",
        protected_variable=protected_attribute_name,
        privileged_num=privileged_num,
        unprivileged_num=unprivileged_num,
    )

    model_summary(f"{protected_attribute_name} fairness metrics", "", original_approach)
    performance_metrics = pd.DataFrame(
        original_approach.items(), columns=["Model Performance Metrics", "Values"]
    )
    plt.show()

    return (
        fun_MLe,
        target_predict,
        pred_continuous,
        score,
        report,
        classifier_df,
        vTest,
        dfM,
        performance_metrics,
        auc_train,
        auc_test,
    )
