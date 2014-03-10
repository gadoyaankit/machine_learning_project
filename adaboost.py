from operator import itemgetter
import math
from collections import  defaultdict

train_data = open('processed_train.txt', 'r')
test_data = open('processed_test.txt', 'r')

number_of_feature = 11

analysis_file = open("analysis_file",'w')

def sign(i):
    if (i < 0):
        return -1
    else:
        return 1
def compute_mcc(fp,tp,fn,tn):

    tptn = tp*tn
    fpfn = fp*fn
    numr = tptn-fpfn
    tpfp = tp+fp
    tpfn = tp+fn
    tnfp=tn+fp
    tnfn=tn+fn
    pden = tpfp*tpfn*tnfp*tnfn
    deno = math.sqrt(pden)
    if deno == 0:
        deno = 1
    mcc = numr/deno
   
    if tp ==0.0:
        tp=1
    if fn ==0.0:
        fn=1
    if fp ==0.0:
        fp=1
    prec = float(tp/(tp+fp))
    recall = float(tp/(tp+fn))
    f1 = 2*(prec*recall)/(prec+recall)
    analysis_file.write(str(mcc)+",")
    analysis_file.write(str(f1)+",")
    print "MCC: "+str(mcc)
    print "F1 score: "+str(f1)

def compute_roc(threshold,test,scores):
    f = open('roc_data','w')

    error = defaultdict(list)
    k=0
    falsePositiveRate = []
    truePositiveRate = []
    
    for  index in threshold:
        truepositive = 0
        falsepositive = 0
        falsenegative = 0 
        truenegative = 0
        for i in range (0,len(scores)): 
            if(scores[i] >= index):
                if(int(test[i][11]) == -1):
                    falsepositive =falsepositive+1
                else:
                    truepositive=truepositive+1
            else:
                if(int(test[i][11]) == 1):
                    falsenegative = falsenegative+1
                else:
                    truenegative = truenegative+1

            i=i+1
        error[k].append(falsepositive)
        error[k].append(truepositive)
        error[k].append(falsenegative)
        error[k].append(truenegative)
        k=k+1
        a = (float(falsepositive)/float(falsepositive+truenegative))
        b = float(truepositive)/float(truepositive+falsenegative)
        falsePositiveRate.append(a)
        truePositiveRate.append(b)

        f.write(str(a)+","+str(b)+"\n")
        
    f.close()
    #computemcc(float(fpn),float(tpn),float(fnn),float(tnn))
    auc = computeauc(falsePositiveRate,truePositiveRate)
    
   
    return auc

def computeauc(xList, yList):
    value = 0
    for i in range(2,5952):
        value += ((float(xList[i]) - float(xList[i - 1])) *  (float(yList[i]) + float(yList[i-1])))
    return float(value) * float(0.5)

def compute_testing_error(featureS, fold):
    testingerror = 0.0
    fp=0.0
    tp=0.0
    tn =0.0
    fn =0.0
    for datapoint in range (0, len(fold)):
        if sign(featureS[datapoint])==1 and fold[datapoint][11]==1:
            tp+=1
        if sign(featureS[datapoint])==1 and fold[datapoint][11]==-1:
            fn+=1
        if sign(featureS[datapoint])==-1 and fold[datapoint][11]==1:
            fp+=1
        if sign(featureS[datapoint])==-1 and fold[datapoint][11]==-1:
            tn+=1
        if(sign(featureS[datapoint]) != float(fold[datapoint][11])):
            testingerror = testingerror + 1 
    acc = (len(fold) - (testingerror)) / len(fold)
    compute_mcc(fp,tp,fn,tn)
    print "Accuracy: " + str(acc)
    testing_error_rate = (testingerror)/ len(fold)
    print "Testing Error: "+str(testing_error_rate)
    
    analysis_file.write(str(testing_error_rate)+",")
    analysis_file.write(str(acc)+",")
    thresholds = sorted(featureS)
    thresholds = reversed(thresholds)
    print tp,fp,tn,fn
    auc = compute_roc(thresholds, fold, featureS)
    return auc


def compute_threshold(temp, feature):
    error = 0.0
    optimalList = []
    thresholdLimit = float(temp[0][feature]) - 1
    
    for datapoint in range (0, len(temp)):
        if(float(temp[datapoint][feature]) > thresholdLimit):
            if(float(temp[datapoint][11]) == -1):
                error += temp[datapoint][12]
        else:
            if(float(temp[datapoint][11]) == 1):
                error += temp[datapoint][12]
                
    optimalList.append([abs(.5 - error), thresholdLimit, feature])
    prev = 0.0
    
    for i in range (0, len(temp)):
        if(i != len(temp) - 1):
            threshold = 0.5 * (temp[i][feature] + temp[i + 1][feature])
        else:
            threshold = (float(temp[i][feature]) + 1) / 2
        if(temp[i][11] == -1):
            error = error - temp[i][12]
        else:
            error = error + temp[i][12]
        if(threshold != prev):
            optimalList.append([abs(.5 - error), error, threshold, feature])
            prev = threshold
            
    thresholds = sorted(optimalList, key=itemgetter(0), reverse=True)

    return feature, thresholds

def computeTrainingFoldError(featureSum,fold):
    trainingerror = 0.0
    for email in range (0,len(fold)):
        if(sign(featureSum[email]) != float(fold[email][11])):
            trainingerror = trainingerror+1 
    
    trainingerror = float(trainingerror/len(fold))
    print str(trainingerror)+","
    analysis_file.write(str(trainingerror)+",")

def Adaboost(train_datapoints, test_datapoints):
    
    
    feature_sum_train = [0.0] * len(train_datapoints)
    feature_sum_test = [0.0] * len(test_datapoints)
    boost_count = 0
    
    while(True):
        optimal_stump = -99999
        temp = []
        sumn = 0.0
        for feature in range(0, 11):
            temp = sorted(train_datapoints, key=itemgetter(feature))
            feature, thresholds = compute_threshold(temp, feature)
            
            if(optimal_stump < thresholds[0][0]):
                optimal_stump = thresholds[0][0]
                error = thresholds[0][1]
                threshold = thresholds[0][2]
                current_feature = feature
            ex_wrong = math.sqrt(float((1 - error) / error))
            ex_right = math.sqrt(float(error / (1 - error)))

        
        log_error = math.log(float((1 - error) / error))
        for datapoint in range (0, len(train_datapoints)):
            
            if(threshold < float(train_datapoints[datapoint][current_feature])):
                feature_sum_train[datapoint] = feature_sum_train[datapoint] + (.5 * log_error) * 1
                if(float(train_datapoints[datapoint][11]) == 1):
                    sumn = sumn + train_datapoints[datapoint][12] * ex_right
                    train_datapoints[datapoint][12] = train_datapoints[datapoint][12] * ex_right
                    
                else:
                    sumn = sumn + train_datapoints[datapoint][12] * ex_wrong
                    train_datapoints[datapoint][12] = train_datapoints[datapoint][12] * ex_wrong
            else:
                feature_sum_train[datapoint] = feature_sum_train[datapoint] + (.5 * log_error) * (-1)
                if(float(train_datapoints[datapoint][11]) == 1):
                    sumn = sumn + train_datapoints[datapoint][12] * ex_wrong
                    train_datapoints[datapoint][12] = train_datapoints[datapoint][12] * ex_wrong
                    
                else:
                    sumn = sumn + train_datapoints[datapoint][12] * ex_right
                    train_datapoints[datapoint][12] = train_datapoints[datapoint][12] * ex_right    
        
        for j in range(0, len(train_datapoints)):
            train_datapoints[j][12] = (train_datapoints[j][12]) / sumn
        computeTrainingFoldError(feature_sum_train,train_datapoints)

        for datapoint in range (0, len(test_datapoints)):
            if(threshold < float(test_datapoints[datapoint][current_feature])):
                feature_sum_test[datapoint] = feature_sum_test[datapoint] + (.5 * log_error) * 1
            else:
                feature_sum_test[datapoint] = feature_sum_test[datapoint] + (.5 * log_error) * (-1)
                
        auc = float(compute_testing_error(feature_sum_test, test_datapoints))
        if auc > 0.985:
            print "Final AUC:" + str(auc)
            print boost_count
            analysis_file.write(str(auc)+",")
            analysis_file.write(str(boost_count)+"\n")
            break
        boost_count += 1
        print "AUC: "+str(auc)

        analysis_file.write(str(auc)+",")
        analysis_file.write(str(boost_count)+"\n")
        print "-------------------------------------------------------------------------------------------------------"


def process_datapoints(train_datapoints, test_datapoints):
    
    length_of_train_data = len(train_datapoints)
    length_of_test_data = len(test_datapoints)
    
    # initial weights
    train_weight = 1.0 / length_of_train_data
    test_weight = 1.0 / length_of_test_data
    
    # Initialised datapoints weights
    # Also replace 0 of the datapoint label with -1
    for i in range(length_of_train_data):
        train_datapoints[i][12] = train_weight
        if(train_datapoints[i][number_of_feature] == 0):
            train_datapoints[i][number_of_feature] = -1
    
    for i in range(length_of_test_data):
        test_datapoints[i][12] = test_weight
        if(test_datapoints[i][number_of_feature] == 0):
            test_datapoints[i][number_of_feature] = -1
    
    Adaboost(train_datapoints, test_datapoints)
            

# ---------------------------------------------------------------------------------------------------------------------    
# Populate data
def populate_data():
    
    # Populate training data
    training_datapoints = []
    for line in train_data:
        line = line.strip("\n").rstrip(" ")
        splitedline = line.split(",")
        
        temp = []
        for i in range(len(splitedline)):
            temp.append(float(splitedline[i]))
        temp.append(0)  # this will be useful for initial weights
        training_datapoints.append(temp)
    
    # Populate testing data
    testing_datapoints = []
    for line in test_data:
        line = line.strip("\n").rstrip(" ")
        splitedline = line.split(",")
    
        temp = []
        for i in range(len(splitedline)):
            temp.append(float(splitedline[i]))
        temp.append(0)
        testing_datapoints.append(temp)
    
    process_datapoints(training_datapoints, testing_datapoints)
    
populate_data()
