import statistics as stat, pandas, numpy as np, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, log_loss, confusion_matrix

# Read in .csv file
data = pandas.read_csv('heart.csv')

# ; 734 instances = training data, 184 instances = test data; 918 total instances
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

# Initialize parameter vector w as a 1xD vector
w = np.ones((1, 5), dtype=float)
# Initialize learning rate alpha
alpha = 0.01
# Define a probability list to store predicted results
probabilityTraining = []
probabilityTest = []

def sigmoid(w):
    return 1 / (1 + np.exp(-w))

def classify(probabilityTraining, probabilityTest, w, x, x2, y, y2):
    for i in range(len(x)):
        temp = w.copy()
        temp[0, 0] -= alpha * ((sigmoid(np.dot(x.iloc[i, :], np.transpose(w))) - y.iloc[i]) * x.iloc[i, 0])
        temp[0, 1] -= alpha * ((sigmoid(np.dot(x.iloc[i, :], np.transpose(w))) - y.iloc[i]) * x.iloc[i, 1])
        temp[0, 2] -= alpha * ((sigmoid(np.dot(x.iloc[i, :], np.transpose(w))) - y.iloc[i]) * x.iloc[i, 2])
        temp[0, 3] -= alpha * ((sigmoid(np.dot(x.iloc[i, :], np.transpose(w))) - y.iloc[i]) * x.iloc[i, 3])
        temp[0, 4] -= alpha * ((sigmoid(np.dot(x.iloc[i, :], np.transpose(w))) - y.iloc[i]) * x.iloc[i, 4])
        probabilityTraining.append(np.asscalar(sigmoid(np.dot(x.iloc[i, :], np.transpose(temp)))))
        if probabilityTraining[i] > 0.5:
            probabilityTraining[i] = 1
        else:
            probabilityTraining[i] = 0

    for i in range(len(x2)):
        temp2 = w.copy()
        temp2[0, 0] -= alpha * ((sigmoid(np.dot(x2.iloc[i, :], np.transpose(w))) - y2.iloc[i]) * x2.iloc[i, 0])
        temp2[0, 1] -= alpha * ((sigmoid(np.dot(x2.iloc[i, :], np.transpose(w))) - y2.iloc[i]) * x2.iloc[i, 1])
        temp2[0, 2] -= alpha * ((sigmoid(np.dot(x2.iloc[i, :], np.transpose(w))) - y2.iloc[i]) * x2.iloc[i, 2])
        temp2[0, 3] -= alpha * ((sigmoid(np.dot(x2.iloc[i, :], np.transpose(w))) - y2.iloc[i]) * x2.iloc[i, 3])
        temp2[0, 4] -= alpha * ((sigmoid(np.dot(x2.iloc[i, :], np.transpose(w))) - y2.iloc[i]) * x2.iloc[i, 4])
        probabilityTest.append(np.asscalar(sigmoid(np.dot(x2.iloc[i, :], np.transpose(temp2)))))
        if probabilityTest[i] > 0.5:
            probabilityTest[i] = 1
        else:
            probabilityTest[i] = 0
    print("Parameter Vector for GD:", temp)

classify(probabilityTraining, probabilityTest, w, TrainingDataInput, TestDataInput, TrainingDataOutput, TestDataOutput)

print("Probability array GD(P(y = 1 | Xi, Θ) == probabilityTraining[i]):", probabilityTraining)
print("Probability array GD(P(y = 1 | Xi, Θ) == probabilityTest[i]):", probabilityTest)
# Evaluation metrics: Accuracy, Error, Sensitivity, Specificity, F1 Score
# Calculate F1 Scores
GDTrainingF1_Score = f1_score(TrainingDataOutput, probabilityTraining)
GDTestF1_Score = f1_score(TestDataOutput, probabilityTest)

# Calculate Accuracy Scores
GDTrainingAccuracy = accuracy_score(TrainingDataOutput, probabilityTraining)
GDTestAccuracy = accuracy_score(TestDataOutput, probabilityTest)

# Calculate Log Loss
GDTrainingLogLoss = log_loss(TrainingDataOutput, probabilityTraining)
GDTestLogLoss = log_loss(TestDataOutput, probabilityTest)

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
GDTrainingError = (fp0 + fn0) / (tp0 + tn0 + fp0 + fn0)
GDTestError = (fp1 + fn1) / (tp1 + tn1 + fp1 + fn1)

# Calculate Sensitivity & Specificity
GDTrainingSensitivity = tp0 / (tp0 + fn0)
GDTrainingSpecificity = tn0 / (tn0 + fp0)
GDTestSensitivity = tp1 / (tp1 + fn1)
GDTestSpecificity = tn1 / (tn1 + fp1)

print("GD Training F1 Score: ", GDTrainingF1_Score)
print("GD Training Accuracy Score: ", GDTrainingAccuracy)
print("GD Training Error: ", GDTrainingError)
print("GD Training Log Loss: ", GDTrainingLogLoss)
print("GD Training Sensitivity: ", GDTrainingSensitivity)
print("GD Training Specificity: ", GDTrainingSpecificity)

print("GD Test F1 Score: ", GDTestF1_Score)
print("GD Test Accuracy Score: ", GDTestAccuracy)
print("GD Test Error: ", GDTestError)
print("GD Test Log Loss: ", GDTestLogLoss)
print("GD Test Sensitivity: ", GDTestSensitivity)
print("GD Test Specificity: ", GDTestSpecificity)

# Initialize parameter vector w as a 1xD vector
wSGD = np.ones((1, 5), dtype=float)
# Define a probability list to store predicted results
probabilityTrainingSGD = []
probabilityTestSGD = []


def classifySGD(probabilityTrainingSGD, probabilityTestSGD, wSGD, x, x2, y, y2):
    for i in range(len(x)):
        temp3 = w.copy()
        temp3[0, 0] -= alpha * ((sigmoid(np.dot(x.iloc[0, :], np.transpose(w))) - y.iloc[i]) * x.iloc[0, 0])
        temp3[0, 1] -= alpha * ((sigmoid(np.dot(x.iloc[0, :], np.transpose(w))) - y.iloc[i]) * x.iloc[0, 1])
        temp3[0, 2] -= alpha * ((sigmoid(np.dot(x.iloc[0, :], np.transpose(w))) - y.iloc[i]) * x.iloc[0, 2])
        temp3[0, 3] -= alpha * ((sigmoid(np.dot(x.iloc[0, :], np.transpose(w))) - y.iloc[i]) * x.iloc[0, 3])
        temp3[0, 4] -= alpha * ((sigmoid(np.dot(x.iloc[0, :], np.transpose(w))) - y.iloc[i]) * x.iloc[0, 4])
        probabilityTrainingSGD.append(np.asscalar(sigmoid(np.dot(x.iloc[i, :], np.transpose(temp3)))))
        if probabilityTrainingSGD[i] > 0.5:
            probabilityTrainingSGD[i] = 1
        else:
            probabilityTrainingSGD[i] = 0

    for i in range(len(x2)):
        temp4 = w.copy()
        temp4[0, 0] -= alpha * ((sigmoid(np.dot(x2.iloc[0, :], np.transpose(w))) - y2.iloc[i]) * x2.iloc[0, 0])
        temp4[0, 1] -= alpha * ((sigmoid(np.dot(x2.iloc[0, :], np.transpose(w))) - y2.iloc[i]) * x2.iloc[0, 1])
        temp4[0, 2] -= alpha * ((sigmoid(np.dot(x2.iloc[0, :], np.transpose(w))) - y2.iloc[i]) * x2.iloc[0, 2])
        temp4[0, 3] -= alpha * ((sigmoid(np.dot(x2.iloc[0, :], np.transpose(w))) - y2.iloc[i]) * x2.iloc[0, 3])
        temp4[0, 4] -= alpha * ((sigmoid(np.dot(x2.iloc[0, :], np.transpose(w))) - y2.iloc[i]) * x2.iloc[0, 4])
        probabilityTestSGD.append(np.asscalar(sigmoid(np.dot(x2.iloc[i, :], np.transpose(temp4)))))
        if probabilityTestSGD[i] > 0.5:
            probabilityTestSGD[i] = 1
        else:
            probabilityTestSGD[i] = 0
    print("Parameter Vector for SGD:", temp3)

classifySGD(probabilityTrainingSGD, probabilityTestSGD, wSGD, TrainingDataInput, TestDataInput, TrainingDataOutput, TestDataOutput)

print("Probability array SGD(P(y = 1 | Xi, Θ) == probabilityTrainingSGD[i]):", probabilityTrainingSGD)
print("Probability array SGD(P(y = 1 | Xi, Θ) == probabilityTestSGD[i]):", probabilityTestSGD)
# Evaluation metrics: Accuracy, Error, Sensitivity, Specificity, F1 Score
# Calculate F1 Scores
SGDTrainingF1_Score = f1_score(TrainingDataOutput, probabilityTrainingSGD)
SGDTestF1_Score = f1_score(TestDataOutput, probabilityTestSGD)

# Calculate Accuracy Scores
SGDTrainingAccuracy = accuracy_score(TrainingDataOutput, probabilityTrainingSGD)
SGDTestAccuracy = accuracy_score(TestDataOutput, probabilityTestSGD)

# Calculate Log Loss
SGDTrainingLogLoss = log_loss(TrainingDataOutput, probabilityTrainingSGD)
SGDTestLogLoss = log_loss(TestDataOutput, probabilityTestSGD)

# Create confusion matrices
SGDTrainingConfusionMatrix = confusion_matrix(TrainingDataOutput, probabilityTrainingSGD)
SGDTestConfusionMatrix = confusion_matrix(TestDataOutput, probabilityTestSGD)
tn0 = SGDTrainingConfusionMatrix[0][0]
fp0 = SGDTrainingConfusionMatrix[0][1]
fn0 = SGDTrainingConfusionMatrix[1][0]
tp0 = SGDTrainingConfusionMatrix[1][1]
tn1 = SGDTestConfusionMatrix[0][0]
fp1 = SGDTestConfusionMatrix[0][1]
fn1 = SGDTestConfusionMatrix[1][0]
tp1 = SGDTestConfusionMatrix[1][1]

# Calculate Error Scores
SGDTrainingError = (fp0 + fn0) / (tp0 + tn0 + fp0 + fn0)
SGDTestError = (fp1 + fn1) / (tp1 + tn1 + fp1 + fn1)

# Calculate Sensitivity & Specificity
SGDTrainingSensitivity = tp0 / (tp0 + fn0)
SGDTrainingSpecificity = tn0 / (tn0 + fp0)
SGDTestSensitivity = tp1 / (tp1 + fn1)
SGDTestSpecificity = tn1 / (tn1 + fp1)

print("SGD Training F1 Score: ", SGDTrainingF1_Score)
print("SGD Training Accuracy Score: ", SGDTrainingAccuracy)
print("SGD Training Error: ", SGDTrainingError)
print("SGD Training Log Loss: ", SGDTrainingLogLoss)
print("SGD Training Sensitivity: ", SGDTrainingSensitivity)
print("SGD Training Specificity: ", SGDTrainingSpecificity)

print("SGD Test F1 Score: ", SGDTestF1_Score)
print("SGD Test Accuracy Score: ", SGDTestAccuracy)
print("SGD Test Error: ", SGDTestError)
print("SGD Test Log Loss: ", SGDTestLogLoss)
print("SGD Test Sensitivity: ", SGDTestSensitivity)
print("SGD Test Specificity: ", SGDTestSpecificity)