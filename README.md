# mortality-project

Predicting Mortality from National Health survey data
Ben Johnson-Laird

For full powerpoint presentation, please see: https://github.com/benlaird/mortality-project/blob/master/Predicting%20Mortality.pptx

Predictors
National Health and Nutrition Examination Survey (NHANES) (2013-2014) 

Dependent Variable
	Mortality (1 or 0) from NCHS Data Linked to NDI Mortality Files (2015)

Survey areas used

Demographic data (e.g. age, race, country of birth)
Laboratory data (e.g. complete blood count, cholesterol)
Questionnaire data (eg. alcohol use, smoking)

Respondents
6,100 participants
   147 were deceased ~1 year later
< 2.5 %

So:  highly unbalanced classes



Exploratory Data Analysis
Initial feature set of ~2,000 features was whittled down to 50 using Random Forest & Feature Importance
The top features by
importance were:

 
Feature
Importance
"Blood urea nitrogen (mg/dL)"
0.0241
Creatinine (umol/L)
0.0170
"Albumin creatinine ratio (mg/g)
0.0113
Blood urea nitrogen (mmol/L)
0.0109
Albumin (g/L)
0.0104
Hydroxycotinine, Serum (ng/mL)
0.0099
Hemoglobin (g/dL)
0.0098
HPV Type 62
0.0077
Direct HDL-Cholesterol (mmol/L)
0.0076

First Model

Enhancements to the initial models

Smote

Grid search result: criterion: 'gini', 'max_depth': 5, 'min_samples_split': 2

Re-smote with Grid search hyper-parameters

Feature set reduction to the top 50 

Models tested

Random Forest

Logistic Regression

XGBoost



Random Forest with Smote & Grid Search hyperparameters

XG  Boost
0.29
0.18
0.22
0.98
0.99
0.99
1.0
0.0
Precision
Recall
F1 score

Logistic Regression

An Idea for Future Work

Predicting mortality should use a training sample of sick respondents and not a training sample of all respondents.

P(death | very sick) >> P(death | well)

Future Step

Use sickness as a continuous predictor and predict mortality probabilistically based on level of sickness








Very
Well
Deceased

Conclusion


With such an unbalanced class for deceased (< 2.5%) Smote only compensates so much

In future, consider using more years of data with differences for the same variable, respondent combination over time

In future, model respondents level of sickness as a predictor and oversample or subsample just the very sick respondents


