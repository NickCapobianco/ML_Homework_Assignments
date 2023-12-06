import statistics as stat, pandas, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Read in .csv file
data = pandas.read_csv('Real_Estate.csv')

# 40 instances = training data; 10 instances = test data; 50 total instances
# Non-continuous input features have been filtered out
InputCharacteristics = data[["TransactionDate", "HouseAge", "DistanceMRT", "NumberConvenience", "Latitude", "Longitude"]]
OutputCharacteristics = data["HousePrice"]

# Split training and test data such that data is not skewed relative to the original data set
TrainingDataInput, TestDataInput, TrainingDataOutput, TestDataOutput \
    = train_test_split(InputCharacteristics, OutputCharacteristics, test_size=0.2, train_size=0.8, shuffle=True)

# Compute the mean for each training input feature
DateMean = stat.mean(TrainingDataInput.loc[:, "TransactionDate"])  # Average Transaction Date
AgeMean = stat.mean(TrainingDataInput.loc[:, "HouseAge"])  # Average House Age
DistanceMean = stat.mean(TrainingDataInput.loc[:, "DistanceMRT"])  # Average Distance to Mass Rapid Transit (MRT)
ConvenienceMean = stat.mean(TrainingDataInput.loc[:, "NumberConvenience"])  # Average number of convenience stores nearby
LatitudeMean = stat.mean(TrainingDataInput.loc[:, "Latitude"])  # Average Latitude
LongitudeMean = stat.mean(TrainingDataInput.loc[:, "Longitude"])  # Average Longitude

# Compute the standard deviation for each training input feature
DateStd = stat.stdev(TrainingDataInput.loc[:, "TransactionDate"])  # Std Dev of Transaction Date
AgeStd = stat.stdev(TrainingDataInput.loc[:, "HouseAge"])  # Std Dev of House Age
DistanceStd = stat.stdev(TrainingDataInput.loc[:, "DistanceMRT"])  # Std Dev of Distance to Mass Rapid Transit (MRT)
ConvenienceStd = stat.stdev(TrainingDataInput.loc[:, "NumberConvenience"])  # Std Dev of number of convenience stores nearby
LatitudeStd = stat.stdev(TrainingDataInput.loc[:, "Latitude"])  # Std Dev of Latitude
LongitudeStd = stat.stdev(TrainingDataInput.loc[:, "Longitude"])  # Std Dev of Longitude

# Create mean and standard deviation lists to utilize in loops
MeanList = [DateMean, AgeMean, DistanceMean, ConvenienceMean, LatitudeMean, LongitudeMean]
StdList = [DateStd, AgeStd, DistanceStd, ConvenienceStd, LatitudeStd, LongitudeStd]

# Standardize the training data m
for i in range(len(MeanList)):
    for j in range(len(TrainingDataInput)):
        TrainingDataInput.iloc[j, i] = (TrainingDataInput.iloc[j, i] - MeanList[i]) / StdList[i]

# Standardize the test data
for i in range(len(MeanList)):
    for j in range(len(TestDataInput)):
        TestDataInput.iloc[j, i] = (TestDataInput.iloc[j, i] - MeanList[i]) / StdList[i]


# Ordinary Least Squares (OLS)
# Getting each parameter for calculations
TrainingDataVector = TrainingDataInput.iloc[0]
TrainingDataTranspose = pandas.DataFrame.transpose(TrainingDataInput)
TrainingDataY = np.dot(TrainingDataVector,TrainingDataTranspose)
TestDataVector = TestDataInput.iloc[0]
TestDataTranspose = pandas.DataFrame.transpose(TestDataInput)
TestDataY = np.dot(TestDataVector,TestDataTranspose)

# Calculating the bias for OLS
TrainingMultiplied = pandas.DataFrame.__matmul__(TrainingDataTranspose,TrainingDataInput) # Numpy ndarray
TrainingBias = np.dot(np.dot(np.linalg.inv(TrainingMultiplied),TrainingDataTranspose),TrainingDataY)
print("Training Bias:", TrainingBias)

TestMultiplied = pandas.DataFrame.__matmul__(TestDataTranspose,TestDataInput) # Numpy ndarray
TestBias = np.dot(np.dot(np.linalg.inv(TestMultiplied),TestDataTranspose),TestDataY)
print("Test Bias:", TestBias)

# Calculating MSE, MAE, R2_Score
TrainingOLSMSE = mean_squared_error(TrainingDataOutput,TrainingDataY)
TestOLSMSE = mean_squared_error(TestDataOutput,TestDataY)
print("Training MSE for OLS:", TrainingOLSMSE)
print("Test MSE for OLS:", TestOLSMSE)

TrainingOLSMAE = mean_absolute_error(TrainingDataOutput,TrainingDataY)
TestOLSMAE = mean_absolute_error(TestDataOutput,TestDataY)
print("Training MAE for OLS:", TrainingOLSMAE)
print("Test MAE for OLS:", TestOLSMAE)

TrainingOLSR2 = r2_score(TrainingDataOutput,TrainingDataY)
TestOLSR2 = r2_score(TestDataOutput,TestDataY)
print("Training R2 for OLS:", TrainingOLSR2)
print("Test R2 for OLS:",TestOLSR2)

# Linear Regression w/ gradient descent
LearningRate = 0.01
TrainingCost = 0
for i in range(0,39):
    for j in range(0,6):
        TrainingCost = pow(TrainingDataY[i] - np.dot(TrainingDataInput.iloc[i],TrainingBias),2) * ((TrainingBias[j] - LearningRate) / 39)

print("Training Cost:", TrainingCost)
