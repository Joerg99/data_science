import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, skew
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.special import boxcox1p
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import xgboost as xgb


train = pd.read_csv(r'data\train.csv')
test = pd.read_csv(r'data\test.csv')

train_id = train['Id']
train.drop('Id', axis=1, inplace=True)
test_id = test['Id']
test.drop('Id', axis=1, inplace=True)


# _, ax = plt.subplots()
# plt.scatter(x = train['GrLivArea'], y = train['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('GrLivArea', fontsize=13)
# plt.show()
train = train.drop(train[train['GrLivArea'] > 4000].index) # löscht Zeilen 

# mu = pos des maximum, sigma = standardabweichung
# mu, sigma = norm.fit(train['SalePrice'])
# print(mu, sigma)
# print(skew(train['SalePrice']))

# plottet Normalverteilung
# sns.distplot(train['SalePrice'] , fit=norm) 
# plt.show() # --> Skewed, daher log norm oder box cox tranformation

# log(1+value) oder boxcox
# train["SalePrice"] = np.log1p(train["SalePrice"])
# train["SalePrice"] = boxcox1p(train["SalePrice"], 0.15)
# print(train['SalePrice'])
# print(skew(train['SalePrice']))


# calculates a best-fit line for the data --> fit=True 
# stats.probplot(train['SalePrice'], plot=plt, fit=True)
# plt.show()


# corrmat = train.corr()
# plt.subplots(figsize=(12,9))
# sns.heatmap(corrmat, vmax=0.9, square=True)
# plt.show()


# impute missing values
def impute_missing_values():
    all_data = pd.concat((train, test)).reset_index(drop=True)
#     all_data.drop('SalePrice', axis=1, inplace=True)
    all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
    all_data["Alley"] = all_data["Alley"].fillna("None")
    all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
    all_data["Fence"] = all_data["Fence"].fillna("None")
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all_data[col] = all_data[col].fillna('None')
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all_data[col] = all_data[col].fillna('None')
    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    all_data = all_data.drop(['Utilities'], axis=1)
    all_data["Functional"] = all_data["Functional"].fillna("Typ")
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
    all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
    return all_data

all_data = impute_missing_values()

# to categorical features: basically int to string conversion
def to_categorical_features():
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)
to_categorical_features()


# string values to int labels
def label_encoding():
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
            'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
            'YrSold', 'MoSold')
    for col in cols:
        lbl = LabelEncoder()
        lbl.fit(list(all_data[col].values))
        all_data[col] = lbl.transform(list(all_data[col].values))
#         all_data[col] = lbl.inverse_transform(all_data[col])
label_encoding()

def unskew_features():
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    # Check the skew of all numerical features
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness = skewness[abs(skewness) > 0.75]

    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        #all_data[feat] += 1
        all_data[feat] = boxcox1p(all_data[feat], lam)
        
#unskew_features()

# dummy variables
all_data = pd.get_dummies(all_data)

train = all_data[:1000] 
print(train.shape)
test = all_data[1000:]
print(test.shape)
y_train = train.SalePrice.values
train.drop('SalePrice', axis=1, inplace=True)

y_test= test.SalePrice.values
test.drop('SalePrice', axis=1, inplace=True)

# Cross Validation
def rmsle_cv(model):
    kf = KFold(n_splits = 5, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

# rmse mit log(saleprice) und unskewed features = 0.04
# rmse mit log(saleprice) und ohne unskewed features = 0.13
# rmse ohne log mit unskewed features= 0.7
# rmse ohne log und ohne unskewed features= 19521 ?????
print(rmsle_cv(model_xgb).mean())

model_xgb.fit(train, y_train)
test_outputs = model_xgb.predict(test)
print(test_outputs[:10])
print(y_test[:10])

####### TODO: Visualisieren wichtigster Features


# Klassen für  
# class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
#     def __init__(self, models):
#         self.models = models
#         
#     def fit(self, X, y):
#         self.models_ = [clone(x) for x in self.models]
#         # Train cloned base models
#         for model in self.models_:
#             model.fit(X, y)
# 
#         return self
#     def predict(self, X):
#         predictions = np.column_stack([model.predict(X) for model in self.models_])
#         return np.mean(predictions, axis=1)   
# 
# class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
#     def __init__(self, base_models, meta_model, n_folds=5):
#         self.base_models = base_models
#         self.meta_model = meta_model
#         self.n_folds = n_folds
#    
#     # We again fit the data on clones of the original models
#     def fit(self, X, y):
#         self.base_models_ = [list() for _ in self.base_models] # _ war mal x
#         self.meta_model_ = clone(self.meta_model)
#         kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
#         
#         # Train cloned base models then create out-of-fold predictions
#         # that are needed to train the cloned meta-model
#         out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
#         for i, model in enumerate(self.base_models):
#             for train_index, holdout_index in kfold.split(X, y):
#                 instance = clone(model)
#                 self.base_models_[i].append(instance)
#                 instance.fit(X[train_index], y[train_index])
#                 y_pred = instance.predict(X[holdout_index])
#                 out_of_fold_predictions[holdout_index, i] = y_pred
#                 
#         # Now train the cloned  meta-model using the out-of-fold predictions as new feature
#         self.meta_model_.fit(out_of_fold_predictions, y)
#         return self
#    
#     #Do the predictions of all base models on the test data and use the averaged predictions as 
#     #meta-features for the final prediction which is done by the meta-model
#     def predict(self, X):
#         meta_features = np.column_stack([
#             np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
#             for base_models in self.base_models_ ])
#         return self.meta_model_.predict(meta_features)
