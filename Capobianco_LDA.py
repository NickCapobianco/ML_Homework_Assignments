import statistics as stat, pandas, numpy as np, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, log_loss, confusion_matrix

# Read in .csv file
data = pandas.read_csv('heart.csv')

# ; 734 instances = training data184 instances = test data; 918 total instances
# Non-continuous input features have been filtered out
InputCharacteristics = data[["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR"]]
OutputCharacteristics = data["HeartDisease"]

# Split training and test data such that data is not skewed relative to the original data set
TrainingDataInput, TestDataInput, TrainingDataOutput, TestDataOutput \
    = train_test_split(InputCharacteristics, OutputCharacteristics, test_size=0.2, train_size=0.8, shuffle=True)
# Compute the mean for each training input feature
AgeMean = stat.mean(TrainingDataInput.loc[:, "Age"])  # Average Age between 53-54
RestingBPMean = stat.mean(TrainingDataInput.loc[:, "RestingBP"])  # Average Resting Blood Pressure between 132-134
CholesterolMean = stat.mean(TrainingDataInput.loc[:, "Cholesterol"])  # Average Cholesterol Levels between 192-202
FastingBSMean = stat.mean(TrainingDataInput.loc[:, "FastingBS"])  # Average Fasting Blood Sugar Levels between 0.23-0.25
MaxHRMean = stat.mean(TrainingDataInput.loc[:, "MaxHR"])  # Average Maximum Heart Rate 135-138

# Compute the standard deviation for each training input feature
AgeStd = stat.stdev(TrainingDataInput.loc[:, "Age"])
RestingBPStd = stat.stdev(TrainingDataInput.loc[:, "RestingBP"])
CholesterolStd = stat.stdev(TrainingDataInput.loc[:, "Cholesterol"])
FastingBSStd = stat.stdev(TrainingDataInput.loc[:, "FastingBS"])
MaxHRStd = stat.stdev(TrainingDataInput.loc[:, "MaxHR"])

# Create mean and standard deviation lists to utilize in loops
MeanList = [AgeMean, RestingBPMean, CholesterolMean, FastingBSMean, MaxHRMean]
StdList = [AgeStd, RestingBPStd, CholesterolStd, FastingBSStd, MaxHRStd]

# Standardize the training data
for i in range(len(MeanList)):
    for j in range(len(TrainingDataInput)):
        TrainingDataInput.iloc[j, i] = (TrainingDataInput.iloc[j, i] - MeanList[i]) / StdList[i]

# Standardize the test data
for i in range(len(MeanList)):
    for j in range(len(TestDataInput)):
        TestDataInput.iloc[j, i] = (TestDataInput.iloc[j, i] - MeanList[i]) / StdList[i]

# Compute the priors and mean for each class in the training data
# Create two iterators to keep track of the number of data instances in each class
class0Count = 0
class1Count = 0

# Create two arrays of indices to reference for each class in the training data
class0ListIndex = []
class1ListIndex = []

# Create two column vectors to append to upon iterating over the training data and extracting class-specific rows
class0Mu = 0
class1Mu = 0

# Incrementing variable for extracting the column vectors for µc of each class
j = 0

for i in TrainingDataOutput:
    if i == 0:
        class0ListIndex.append(j)
        class0Mu += TrainingDataInput.iloc[j, :]
        class0Count += 1
    elif i == 1:
        class1ListIndex.append(j)
        class1Mu += TrainingDataInput.iloc[j, :]
        class1Count += 1
    j += 1

# Prior calculations for each class in the training data
prior0 = class0Count / 734  # Prior0 is indicative of the probabilityTest that one does not have a Heart Disease
prior1 = class1Count / 734  # Prior1 is indicative of the probabilityTest that one has a Heart Disease
class0Mu = class0Mu / class0Count
class1Mu = class1Mu / class1Count

# Calculating (xi-µc) for each sigma class
# Iterate through each row in the training data and subtract the respective mean from the proper input feature
for j in class0ListIndex:
    TrainingDataInput.iloc[j, :] = TrainingDataInput.iloc[j, :] - class0Mu
for k in class1ListIndex:
    TrainingDataInput.iloc[k, :] = TrainingDataInput.iloc[k, :] - class1Mu

# Create separate matrices for each class after subtracting µc
class0MatrixForSigma = TrainingDataInput.iloc[class0ListIndex, :]
class1MatrixForSigma = TrainingDataInput.iloc[class1ListIndex, :]

# Calculate each sigma summation by class
class0MatrixForSigma = pandas.DataFrame.__matmul__(pandas.DataFrame.transpose(class0MatrixForSigma),class0MatrixForSigma)
class1MatrixForSigma = pandas.DataFrame.__matmul__(pandas.DataFrame.transpose(class1MatrixForSigma),class1MatrixForSigma)
class0MatrixForSigma = class0MatrixForSigma / class0Count
class1MatrixForSigma = class1MatrixForSigma / class1Count
time.sleep(0.5)

# Calculate sigma_pooled and estimate the shared covariance
# sigmaPooled = (((class0Count - 1) * class0MatrixForSigma) / (class0Count - 1)) + (((class1Count - 1) * class1MatrixForSigma) / (class1Count - 1))
sigmaPooled = (((class0Count - 1) * class0MatrixForSigma) + ((class1Count - 1) * class1MatrixForSigma)) / (
        (class0Count - 1) + (class1Count - 1))
print("Sigma Pooled:\n", sigmaPooled)
print("Class0, Class1:", class0Count, class1Count)
print("Sigma Class 0:", class0MatrixForSigma)
print("Sigma Class 1:", class1MatrixForSigma)
time.sleep(0.5)

# # Calculate beta (inverse of shared covariance matrix * class means)
print("Sigma Pooled Inverse:", np.linalg.inv(sigmaPooled))
beta0 = np.dot(np.linalg.inv(sigmaPooled), class0Mu)
beta1 = np.dot(np.linalg.inv(sigmaPooled), class1Mu)
print("Beta0:", beta0)
print("Beta1:", beta1)

# # Calculate gamma for each input feature by class
# Class 0
gamma0 = (-1 / 2) * np.dot(np.transpose(class0Mu), beta0) + np.log(prior0)
gamma1 = (-1 / 2) * np.dot(np.transpose(class1Mu), beta1) + np.log(prior1)
print("gamma0:", gamma0)
print("gamma1:", gamma1)
time.sleep(0.5)

# Calculate beta and gamma differentials
betaDifference = beta1 - beta0
gammaDifference = gamma1 - gamma0
betaDifference = np.transpose(np.array([betaDifference]))
print("Beta Difference:", betaDifference)
print("Gamma Difference:", gammaDifference)
time.sleep(0.5)


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Calculate the probabilityTest for each data instance in the test data set and assign it to a class
probabilityTest = []
probabilityTraining = []

for i in range(len(TrainingDataInput)):
    probabilityTraining.append(np.asscalar(sigmoid(np.dot(np.transpose(betaDifference), TrainingDataInput.iloc[i, :]) + gammaDifference)))
    if probabilityTraining[i] >= 0.5:
        probabilityTraining[i] = 1
    else:
        probabilityTraining[i] = 0

for i in range(len(TestDataInput)):
    probabilityTest.append(np.asscalar(sigmoid(np.dot(np.transpose(betaDifference), TestDataInput.iloc[i, :]) + gammaDifference)))
    if probabilityTest[i] >= 0.5:
        probabilityTest[i] = 1
    else:
        probabilityTest[i] = 0

print("Probability array(P(y = 1 | Xi, Θ) == probabilityTraining[i]):", probabilityTraining)
print("Probability array(P(y = 1 | Xi, Θ) == probabilityTest[i]):", probabilityTest)
# Evaluation metrics: Accuracy, Error, Sensitivity, Specificity, Precision, F1 Score
# Calculate F1 Scores
TrainingF1_Score = f1_score(TrainingDataOutput, probabilityTraining)
TestF1_Score = f1_score(TestDataOutput, probabilityTest)

# Calculate Accuracy Scores
TrainingAccuracy = accuracy_score(TrainingDataOutput, probabilityTraining)
TestAccuracy = accuracy_score(TestDataOutput, probabilityTest)

# Calculate Log Loss
TrainingLogLoss = log_loss(TrainingDataOutput, probabilityTraining)
TestLogLoss = log_loss(TestDataOutput, probabilityTest)

# Create confusion matrices
TrainingConfusionMatrix = confusion_matrix(TrainingDataOutput, probabilityTraining)
TestConfusionMatrix = confusion_matrix(TestDataOutput, probabilityTest)
tn0 = TrainingConfusionMatrix[0][0]
fp0 = TrainingConfusionMatrix[0][1]
fn0 = TrainingConfusionMatrix[1][0]
tp0 = TrainingConfusionMatrix[1][1]
tn1 = TestConfusionMatrix[0][0]
fp1 = TestConfusionMatrix[0][1]
fn1 = TestConfusionMatrix[1][0]
tp1 = TestConfusionMatrix[1][1]

# Calculate Error Scores
TrainingError = (fp0 + fn0) / (tp0 + tn0 + fp0 + fn0)
TestError = (fp1 + fn1) / (tp1 + tn1 + fp1 + fn1)

# Calculate Sensitivity & Specificity
TrainingSensitivity = tp1 / (tp1 + fn1)
TrainingSpecificity = tn1 / (tn1 + fp1)
TestSensitivity = tp1 / (tp1 + fn1)
TestSpecificity = tn1 / (tn1 + fp1)

print("Training F1 Score: ", TrainingF1_Score)
print("Training Accuracy Score: ", TrainingAccuracy)
print("Training Error: ", TrainingError)
print("Training Log Loss: ", TrainingLogLoss)
print("Training Sensitivity: ", TrainingSensitivity)
print("Training Specificity: ", TrainingSpecificity)

print("Test F1 Score: ", TestF1_Score)
print("Test Accuracy Score: ", TestAccuracy)
print("Test Error: ", TestError)
print("Test Log Loss: ", TestLogLoss)
print("Test Sensitivity: ", TestSensitivity)
print("Test Specificity: ", TestSpecificity)