import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression,LogisticRegression
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag


A1 = pd.read_excel("results.xlsx", sheet_name = 'A1',engine='openpyxl')
A2 = pd.read_excel("results.xlsx", sheet_name = 'A2',engine='openpyxl')
A3 = pd.read_excel("results.xlsx", sheet_name = 'A1',engine='openpyxl')
A4 = pd.read_excel("results.xlsx", sheet_name = 'A2',engine='openpyxl')

B1 = pd.read_excel("results.xlsx", sheet_name = 'B1',engine='openpyxl')
B2 = pd.read_excel("results.xlsx", sheet_name = 'B2',engine='openpyxl')
B3 = pd.read_excel("results.xlsx", sheet_name = 'B3',engine='openpyxl')
B4 = pd.read_excel("results.xlsx", sheet_name = 'B4',engine='openpyxl')

C1 = pd.read_excel("results.xlsx", sheet_name = 'C1',engine='openpyxl')
C2 = pd.read_excel("results.xlsx", sheet_name = 'C2',engine='openpyxl')
C3 = pd.read_excel("results.xlsx", sheet_name = 'C3',engine='openpyxl')
C4 = pd.read_excel("results.xlsx", sheet_name = 'C4',engine='openpyxl')

D1 = pd.read_excel("results.xlsx", sheet_name = 'D1',engine='openpyxl')
D2 = pd.read_excel("results.xlsx", sheet_name = 'D2',engine='openpyxl')
D3 = pd.read_excel("results.xlsx", sheet_name = 'D3',engine='openpyxl')
D4 = pd.read_excel("results.xlsx", sheet_name = 'D4',engine='openpyxl')

# A = screen, smoke
# B = voice, smoke
# C = screen, no smoke
# D = voice, no smoke

def preprocess(file, scenario):
        if scenario == "A":
            file['smoke'] = 1
            file['scenario'] = 1#'screen'
        elif scenario == "B":
            file['smoke'] = 1
            file['scenario'] = 0#'voice'
        elif scenario == "C":
            file['smoke'] = 0
            file['scenario'] = 1#'screen'
        else:
            file['smoke'] = 0
            file['scenario'] = 0#'voice'
        return file


A1 = preprocess(A1, "A")
A2 = preprocess(A2, "A")
A3 = preprocess(A3, "A")
A4 = preprocess(A4, "A")
B1 = preprocess(B1, "B")
B2 = preprocess(B2, "B")
B3 = preprocess(B3, "B")
B4 = preprocess(B4, "B")
C1 = preprocess(C1, "C")
C2 = preprocess(C2, "C")
C3 = preprocess(C3, "C")
C4 = preprocess(C4, "C")
D1 = preprocess(D1, "D")
D2 = preprocess(D2, "D")
D3 = preprocess(D3, "D")
D4 = preprocess(D4, "D")

A = pd.DataFrame()
A['time'] = A1['time'] + A2['time'] + A3['time'] + A4['time']
# A2['time'] = A1['time'] + A2['time'] + A3['time'] + A4['time']
# A3['time'] = A1['time'] + A2['time'] + A3['time'] + A4['time']
# A4['time'] = A1['time'] + A2['time'] + A3['time'] + A4['time']


print("A avg:")
print(np.mean(A['time']))
print("A sd:")
print(np.std(A['time']))

B = pd.DataFrame()
B['time'] = B1['time'] + B2['time'] + B3['time'] + B4['time']
# A2['time'] = A1['time'] + A2['time'] + A3['time'] + A4['time']
# A3['time'] = A1['time'] + A2['time'] + A3['time'] + A4['time']
# A4['time'] = A1['time'] + A2['time'] + A3['time'] + A4['time']


print("B avg:")
print(np.mean(B['time']))
print("A sd:")
print(np.std(B['time']))


C = pd.DataFrame()
C['time'] = C1['time'] + C2['time'] + C3['time'] + C4['time']
# A2['time'] = A1['time'] + A2['time'] + A3['time'] + A4['time']
# A3['time'] = A1['time'] + A2['time'] + A3['time'] + A4['time']
# A4['time'] = A1['time'] + A2['time'] + A3['time'] + A4['time']


print("C avg:")
print(np.mean(C['time']))
print("C sd:")
print(np.std(C['time']))


D = pd.DataFrame()
D['time'] = D1['time'] + D2['time'] + D3['time'] + D4['time']
# A2['time'] = A1['time'] + A2['time'] + A3['time'] + A4['time']
# A3['time'] = A1['time'] + A2['time'] + A3['time'] + A4['time']
# A4['time'] = A1['time'] + A2['time'] + A3['time'] + A4['time']


print("D avg:")
print(np.mean(D['time']))
print("D sd:")
print(np.std(D['time']))

# level1 = [A1,B1,C1,D1]
# level1 = [A1,C1]
level1 = [B1,D1]
level1 = pd.concat(level1)
# level1 = pd.get_dummies(level1)
# level1= level1[(np.abs(stats.zscore(level1['time'])) < 3)]
print("level1")
print(stats.pointbiserialr(level1['smoke'], level1['time']))
print(stats.pointbiserialr(level1['scenario'], level1['time']))
# plt.plot(level1['scenario'], level1['time'])
# plt.show()

# level2 = [A2,B2,C2,D2]
# level2 = [A2,C2]
level2 = [B2,D2]
level2 = pd.concat(level2)
print("level2")
print(stats.pointbiserialr(level2['smoke'], level2['time']))
print(stats.pointbiserialr(level2['scenario'], level2['time']))
# level2 = pd.get_dummies(level2)
# level2 = level2[(np.abs(stats.zscore(level2['time'])) < 3)]

# level3 = [A3,B3,C3,D3]
# level3 = [A3,C3]
level3 = [B3,D3]
level3 = pd.concat(level3)
print("level3")
print(stats.pointbiserialr(level3['smoke'], level3['time']))
print(stats.pointbiserialr(level3['scenario'], level3['time']))
# level3 = pd.get_dummies(level3)
# level3 = level3[(np.abs(stats.zscore(level3['time'])) < 3)]

# level4 = [A4,B4,C4,D4]
# level4 = [A4,C4]
level4 = [B4,D4]
level4 = pd.concat(level4)
print("level4")
print(stats.pointbiserialr(level4['smoke'], level4['time']))
print(stats.pointbiserialr(level4['scenario'], level4['time']))
# level4 = pd.get_dummies(level4)
# level4 = level4[(np.abs(stats.zscore(level4['time'])) < 3)]

# temp1 = pd.DataFrame()
# temp2 = pd.DataFrame()
# temp3 = pd.DataFrame()
# temp4 = pd.DataFrame()
# temp1['time']   = level1['time']
# temp2['time']   = level2['time']
# temp3['time']   = level3['time']
# temp4['time']   = level4['time']

level1['time'] = level1['time'] + level2['time'] + level3['time'] + level4['time']
# level2['time'] = level1['time'] + level2['time'] + level3['time'] + level4['time'] - temp2['time'] - temp3['time'] - temp4['time']
# level3['time'] = level1['time'] + level2['time'] + level3['time'] + level4['time'] - temp1['time'] - temp3['time'] - temp4['time']
# level4['time'] = level1['time'] + level2['time'] + level3['time'] + level4['time'] - temp1['time'] - temp2['time'] - temp3['time']
# allLevels = [level1,level2,level3,level4]
# allLevels = pd.concat(allLevels)
print("all levels")
print(stats.pointbiserialr(level1['smoke'], level1['time']))
print(stats.pointbiserialr(level1['scenario'], level1['time']))

def fitRegression(file):
    linear_y = file['time']
    linear_X = file.drop(["time"],axis=1)
    linear_model = LinearRegression()
    linear_model.fit(linear_X, linear_y)
    linear_r2 = linear_model.score(linear_X, linear_y)
    print('R^2: {0}'.format(linear_r2))
    return linear_model, linear_X, linear_y


def fitLogisticRegression(file):
    linear_y = file['correct']
    linear_X = file.drop(["time"],axis=1)
    linear_X = linear_X.drop(["correct"], axis=1)
    linear_model = LogisticRegression()
    linear_model.fit(linear_X, linear_y)
    return linear_model, linear_X, linear_y


def calculate_residuals(model, features, label):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])

    return df_results


def linear_assumption(model, features, label):
    """
    Linearity: Assumes that there is a linear relationship between the predictors and
               the response variable. If not, either a quadratic term or another
               algorithm should be used.
    """
    print('Assumption 1: Linear Relationship between the Target and the Feature', '\n')

    print('Checking with a scatter plot of actual vs. predicted.',
          'Predictions should follow the diagonal line.')

    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)

    # Plotting the actual vs predicted values
    sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, size=7)

    # Plotting the diagonal line
    line_coords = np.arange(df_results.min().min(), df_results.max().max())
    plt.plot(line_coords, line_coords,  # X and y points
             color='darkorange', linestyle='--')
    plt.title('Actual vs. Predicted')
    plt.show()


def homoscedasticity_assumption(model, features, label):
    """
    Homoscedasticity: Assumes that the errors exhibit constant variance
    """
    print('Assumption 5: Homoscedasticity of Error Terms', '\n')

    print('Residuals should have relative constant variance')

    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)

    # Plotting the residuals
    plt.subplots(figsize=(12, 6))
    ax = plt.subplot(111)  # To remove spines
    plt.scatter(x=df_results.index, y=df_results.Residuals, alpha=0.5)
    plt.plot(np.repeat(0, df_results.index.max()), color='darkorange', linestyle='--')
    ax.spines['right'].set_visible(False)  # Removing the right spine
    ax.spines['top'].set_visible(False)  # Removing the top spine
    plt.title('Residuals')

    plt.show()
    print(min(diag.acorr_ljungbox(df_results.Residuals, lags=40)[1]))

def normal_errors_assumption(model, features, label, p_value_thresh=0.05):
    """
    Normality: Assumes that the error terms are normally distributed. If they are not,
    nonlinear transformations of variables may solve this.

    This assumption being violated primarily causes issues with the confidence intervals
    """
    from statsmodels.stats.diagnostic import normal_ad
    print('Assumption 2: The error terms are normally distributed', '\n')

    # Calculating residuals for the Anderson-Darling test
    df_results = calculate_residuals(model, features, label)

    print('Using the Anderson-Darling test for normal distribution')

    # Performing the test on the residuals
    p_value = normal_ad(df_results['Residuals'])[1]
    print('p-value from the test - below 0.05 generally means non-normal:', p_value)

    # Reporting the normality of the residuals
    if p_value < p_value_thresh:
        print('Residuals are not normally distributed')
    else:
        print('Residuals are normally distributed')

    # Plotting the residuals distribution
    plt.subplots(figsize=(12, 6))
    plt.title('Distribution of Residuals')
    sns.distplot(df_results['Residuals'])
    plt.show()

    print()
    if p_value > p_value_thresh:
        print('Assumption satisfied')
    else:
        print('Assumption not satisfied')
        print()
        print('Confidence intervals will likely be affected')
        print('Try performing nonlinear transformations on variables')


# plt.hist(level1)
# plt.show()
#
# level1['time'] = np.log(level1['time'])
# sns.distplot(level1['time'])
# fig = plt.figure()
# plt.show()
#
# plt.hist(level1)
# plt.show()
#
# plt.hist(level2)
# plt.show()
#
# plt.hist(level3)
# plt.show()
#
# plt.hist(level4)
# plt.show()




# # level1['time'] = np.log(level1['time'])

# linear_model, linear_X, linear_y = fitRegression(allLevels)
# results = sm.OLS(linear_y, linear_X).fit()
# print(results.summary())
#
# allLevels = [level1,level2,level3,level4]
# allLevels = pd.concat(allLevels)
# linear_model, linear_X, linear_y = fitLogisticRegression(allLevels)
# results = sm.Logit(linear_y, linear_X).fit()
# print(results.summary())

# # linear_assumption(linear_model, linear_X, linear_y)
# # homoscedasticity_assumption(linear_model, linear_X, linear_y)
# # normal_errors_assumption(linear_model, linear_X, linear_y)
# #
# # # level2['time'] = np.log(level2['time'])
# linear_model, linear_X, linear_y = fitRegression(level2)
# results = sm.OLS(linear_y, linear_X).fit()
# print(results.summary())
# # linear_assumption(linear_model, linear_X, linear_y)
# # homoscedasticity_assumption(linear_model, linear_X, linear_y)
# # normal_errors_assumption(linear_model, linear_X, linear_y)
# #
# # # level3['time'] = np.log(level3['time'])
# linear_model, linear_X, linear_y = fitRegression(level3)
# results = sm.OLS(linear_y, linear_X).fit()
# print(results.summary())
# # linear_assumption(linear_model, linear_X, linear_y)
# # homoscedasticity_assumption(linear_model, linear_X, linear_y)
# # normal_errors_assumption(linear_model, linear_X, linear_y)
#
# # level4['time'] = np.log(level4['time'])
#
# linear_model, linear_X, linear_y = fitRegression(level4)
# results = sm.OLS(linear_y, linear_X).fit()
# print(results.summary())
# # linear_assumption(linear_model, linear_X, linear_y)
# # homoscedasticity_assumption(linear_model, linear_X, linear_y)
# # normal_errors_assumption(linear_model, linear_X, linear_y)

# level1['time'] = level1['time'] + level2['time'] + level3['time'] + level4['time']
# allLevels = [level1,level2,level3,level4]
# allLevels = pd.concat(allLevels)
# print("all levels")
# print(stats.pointbiserialr(allLevels['smoke'], allLevels['time']))
# print(stats.pointbiserialr(allLevels['scenario'], allLevels['time']))
# # allLevels = allLevels.drop(["correct"], axis=1)
# # allLevels['time'] = np.log(allLevels['time'])
#
# level1 = level1.drop(["correct"], axis=1)
# print("summ of all levels")
# print(stats.pointbiserialr(level1['smoke'], level1['time']))
# print(stats.pointbiserialr(level1['scenario'], level1['time']))
# # level1['time'] = np.log(level1['time'])
# level1 = level1.drop(["smoke"], axis=1)
# linear_model, linear_X, linear_y = fitRegression(level1)
# linear_assumption(linear_model, linear_X, linear_y)
# homoscedasticity_assumption(linear_model, linear_X, linear_y)
# normal_errors_assumption(linear_model, linear_X, linear_y)
#
# results = sm.OLS(linear_y, linear_X).fit()
# print(results.summary())
#
#
# level2 = level2.drop(["correct"], axis=1)
# print(stats.pointbiserialr(level2['smoke'], level2['time']))
# print(stats.pointbiserialr(level2['scenario'], level2['time']))
# # level1['time'] = np.log(level1['time'])
# level2 = level2.drop(["smoke"], axis=1)
# linear_model, linear_X, linear_y = fitRegression(level2)
# linear_assumption(linear_model, linear_X, linear_y)
# homoscedasticity_assumption(linear_model, linear_X, linear_y)
# normal_errors_assumption(linear_model, linear_X, linear_y)
#
# results = sm.OLS(linear_y, linear_X).fit()
# print(results.summary())