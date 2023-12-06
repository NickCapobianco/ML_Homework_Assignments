import statistics as stat, pandas, numpy as np, time, random as r
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, log_loss, confusion_matrix

# Read in .csv file
data = pandas.read_csv('heart.csv')

# 734 instances = training data; 184 instances = test data; 918 total instances
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

# Define the input layers with a bias (x_n)
TrainingX = np.ones([len(TrainingDataInput), 6], dtype=float)
TestX = np.ones([len(TestDataInput), 6], dtype=float)

# Declare weights (v,w) with uniform distribution
v = np.random.uniform(low=-1, high=1, size=(6, 6))
w = np.random.uniform(low=-1, high=1, size=(1, 7))

# Load Training Data Input values to the Training input layer
for i in range(len(TrainingX)):
    TrainingX[i, 1] = TrainingDataInput.iloc[i, 0]
    TrainingX[i, 2] = TrainingDataInput.iloc[i, 1]
    TrainingX[i, 3] = TrainingDataInput.iloc[i, 2]
    TrainingX[i, 4] = TrainingDataInput.iloc[i, 3]
    TrainingX[i, 5] = TrainingDataInput.iloc[i, 4]

# Load Test Data Input values to the Test input layer
for i in range(len(TestX)):
    TestX[i, 1] = TestDataInput.iloc[i, 0]
    TestX[i, 2] = TestDataInput.iloc[i, 1]
    TestX[i, 3] = TestDataInput.iloc[i, 2]
    TestX[i, 4] = TestDataInput.iloc[i, 3]
    TestX[i, 5] = TestDataInput.iloc[i, 4]

# Instantiate error variables for convergence check
InitialErrorV = np.zeros([len(TrainingDataInput),6], dtype = float)
InitialErrorW = np.zeros([len(TrainingDataInput),6], dtype = float)

#Counting epochs
EpochCount = 0

# Recursive function to train my model
def ModelTraining(TrainingX, TestX, v, w, InitialErrorV, InitialErrorW, EpochCount):
    # Calculating Pre-synaptic values for Training and Test Data hidden layer (A)
    # Creating the hidden layer for Training + Test data (hidden units[6] + bias[1] = 7)
    TrainingA = np.dot(TrainingX.copy(), np.transpose(v))
    TestA = np.dot(TestX.copy(), np.transpose(v))
    # sigmoid function (g)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Calculate post-synaptic values for Training and Test data at the hidden layer (Z)
    TrainingOutputZ = sigmoid(TrainingA.copy())
    TestOutputZ = sigmoid(TestA.copy())
    TrainingOutputZAppended = np.ones([len(TrainingOutputZ), 7], dtype = float)
    TestOutputZAppended = np.ones([len(TestOutputZ), 7], dtype = float)

    for i in range(len(TrainingOutputZ)):
        TrainingOutputZAppended[i, 1] = TrainingOutputZ[i, 0]
        TrainingOutputZAppended[i, 2] = TrainingOutputZ[i, 1]
        TrainingOutputZAppended[i, 3] = TrainingOutputZ[i, 2]
        TrainingOutputZAppended[i, 4] = TrainingOutputZ[i, 3]
        TrainingOutputZAppended[i, 5] = TrainingOutputZ[i, 4]
        TrainingOutputZAppended[i, 6] = TrainingOutputZ[i, 5]
    for i in range(len(TestOutputZ)):
        TestOutputZAppended[i, 1] = TestOutputZ[i, 0]
        TestOutputZAppended[i, 2] = TestOutputZ[i, 1]
        TestOutputZAppended[i, 3] = TestOutputZ[i, 2]
        TestOutputZAppended[i, 4] = TestOutputZ[i, 3]
        TestOutputZAppended[i, 5] = TestOutputZ[i, 4]
        TestOutputZAppended[i, 6] = TestOutputZ[i, 5]


    # Create the pre-synaptic variables for Training and Test data at the output layer (B)
    TrainingOutputB = np.dot(TrainingOutputZAppended.copy(), np.transpose(w))
    TestOutputB = np.dot(TestOutputZAppended.copy(), np.transpose(w))

    # Calculate the post-synaptic values for Training and Test data at the output layer (Y^)
    # Computing the error signal vectors at the output layer (Delta_w)
    TrainingY_hat = sigmoid(TrainingOutputB.copy())
    TestY_hat = sigmoid(TestOutputB.copy())
    TrainingDelta_w = TrainingY_hat.copy()
    TestDelta_w = TestY_hat.copy()
    TrainingDataOutputCopy = TrainingDataOutput.copy()
    TestDataOutputCopy = TestDataOutput.copy()

    for i in range(len(TrainingDelta_w)):
        TrainingDelta_w[i] -= TrainingDataOutputCopy.iloc[i]
    for i in range(len(TestDelta_w)):
        TestDelta_w[i] -= TestDataOutputCopy.iloc[i]

    # Calculate the error signal vectors at the input layer (Delta_v)
    TrainingDelta_v_temp = np.dot(TrainingDelta_w, w)
    TestDelta_v_temp = np.dot(TestDelta_w, w)

    # Summation portion (Z)
    TrainingZMultiplication = TrainingOutputZAppended.copy()
    TestZMultiplication = TestOutputZAppended.copy()

    # Subtraction portion (1-Z) element-wise
    TrainingZSubtraction= np.ones([len(TrainingOutputZ), 7], dtype = float)
    TestZSubtraction = np.ones([len(TestOutputZ), 7], dtype = float)
    for i in range(len(TrainingZSubtraction)):
        for j in range(0,7):
            TrainingZSubtraction[i,j] -= TrainingZMultiplication[i,j]
    for i in range(len(TestZSubtraction)):
        for j in range(0, 7):
            TestZSubtraction[i, j] -= TestZMultiplication[i, j]

    # Z * (1-Z) multiplication element-wise
    for i in range(len(TrainingZMultiplication)):
        for j in range(0, 7):
            TrainingZMultiplication[i, j] *= TrainingZSubtraction[i, j]
    for i in range(len(TestZMultiplication)):
        for j in range(0, 7):
            TestZMultiplication[i, j] *= TestZSubtraction[i, j]

    # Multiplying everything together to form Delta_v
    for i in range(len(TrainingZMultiplication)):
        for j in range(0, 7):
            TrainingDelta_v_temp[i, j] *= TrainingZMultiplication[i, j]
    for i in range(len(TestZMultiplication)):
        for j in range(0, 7):
            TestDelta_v_temp[i, j] *= TestZMultiplication[i, j]

    # Remove the 0th column from my v weight vector
    TrainingDelta_v = np.ones([len(TrainingDelta_v_temp), 6], dtype=float)
    TestDelta_v = np.ones([len(TestDelta_v_temp), 6], dtype=float)
    for i in range(len(TrainingDelta_v)):
        for j in range(0, 6):
            TrainingDelta_v[i, j] = TrainingDelta_v_temp[i, j + 1]
    for i in range(len(TestDelta_v)):
        for j in range(0, 6):
            TestDelta_v[i, j] = TestDelta_v_temp[i, j + 1]

    # Calculating the gradients
    TrainingW_Gradient = np.dot(np.transpose(TrainingDelta_w), TrainingOutputZAppended.copy())
    TestW_Gradient = np.dot(np.transpose(TestDelta_w), TestOutputZAppended.copy())
    TrainingV_Gradient = np.dot(np.transpose(TrainingDelta_v), TrainingX.copy())
    TestV_Gradient = np.dot(np.transpose(TestDelta_v), TestX.copy())


    # Updating the weights and introducing a learning rate
    learningRate = 0.003
    DifferenceV = np.ones([len(TrainingDelta_v),6], dtype = float)
    DifferenceW = np.ones([len(TrainingDelta_v),6], dtype = float)
    w = w - (learningRate * (1 / len(TrainingX))) * TrainingW_Gradient
    v = v - (learningRate * (1 / len(TrainingX))) * TrainingV_Gradient

    # Error signal check
    for i in range(len(DifferenceV)):
        for j in range(0,6):
            DifferenceV[i, j] = TrainingDelta_v[i, j] - InitialErrorV[i, j]
            if (DifferenceV[i, j] <= 0.001):
                break
            else:
                EpochCount += 1
                ModelTraining(TrainingX, TestX, v, w, TrainingDelta_v, TrainingDelta_w, EpochCount)

    return EpochCount, TrainingY_hat, TestY_hat

# Evaluation metrics: Accuracy, Error, Sensitivity, Specificity, Precision, F1 Score
# # Calculate the probability for each data instance in the Training and Test datasets and assign them to a class
probabilityTraining = []
probabilityTest = []
EpochCount, TrainingY_hat, TestY_hat = ModelTraining(TrainingX, TestX, v, w, InitialErrorV, InitialErrorW, EpochCount)

for i in range(len(TrainingX)):
    probabilityTraining.append(np.asscalar(TrainingY_hat[i]))
    if (TrainingY_hat[i] >= 0.5):
        probabilityTraining[i] = 1
    else:
        probabilityTraining[i] = 0

for i in range(len(TestX)):
    probabilityTest.append(np.asscalar(TestY_hat[i]))
    if (TestY_hat[i] >= 0.5):
        probabilityTest[i] = 1
    else:
        probabilityTest[i] = 0

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
TrainingSensitivity = tp0 / (tp0 + fn0)
TrainingSpecificity = tn0 / (tn0 + fp0)
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

print("Number of epochs to converge: ", EpochCount)