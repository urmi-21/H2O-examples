##LAB2
setwd("~/work/course/cs573/lab2")
library(h2o)
library(data.table)
train <- fread("optdigits.tra",stringsAsFactors = T,colClasses = c(rep('numeric',64),'character'))
test<-fread("optdigits.tes",stringsAsFactors = T,colClasses = c(rep('numeric',64),'character'))
#Divide into train and test/validation
c.train <- train[1:(nrow(train)*.8),]
c.validation  <- train[(nrow(train)*.2):nrow(train),]
h2o.init(nthreads=-1, enable_assertions = FALSE)
h2o.no_progress()
#data to h2o cluster
train.h2o <- as.h2o(c.train)
validation.h2o <- as.h2o(c.validation)
test.h2o <- as.h2o(test)
#last variable is the class category
y.dep<-"V65"
predictors<-setdiff(names(c.train), y.dep)
######Set model hyper-parameters
hiddenLayers<-c(1,2,3)
hiddenUnits<-c(100,200,300)
learningRates<-c(0.005,0.01)
momentumStart<-c(0,0.5)
inputScaling<-c(T,F)
errorFunc<-c("Quadratic","CrossEntropy")
#hiddenLayers<-c(1,2)
#hiddenUnits<-c(100,200)
#learningRates<-c(0.01)
#momentumStart<-c(0)
#inputScaling<-c(T)
#errorFunc<-c("Quadratic")
##Table to write results
header1<-c("ErrorFunction","layers","hiddenUnits","learnRate","momentumStart","Scale","Acc_val","Acc_test","Time in min")
resultsTab<-data.frame(matrix(ncol = 9, nrow = 0))
colnames(resultsTab)<-header1
#set seed for reproducible results
set.seed(1263)
#save the best model i.e. highest accuraccy on test set
bestModel<-NULL
bestAcc<-0
#make all comniations of the parameters
for(errF in errorFunc){
  for(hL in hiddenLayers){
    for(hU in hiddenUnits){
      for(lR in learningRates){
        for(mS in momentumStart){
          for(iS in inputScaling){
            
            #build model and calculate accuraccy
            s <- proc.time() #start time
            model1<- h2o.deeplearning(x=predictors,y = y.dep,training_frame = train.h2o, validation_frame=validation.h2o,hidden = c(rep(hU),hL), activation = "Rectifier",epochs = 100,loss=errF,rate=lR,momentum_start = mS,standardize = iS,adaptive_rate=F)
            d <- proc.time()  - s #end time
            print("Model training metrics")
            model1@model$training_metrics
            print("Model validation metrics")
            model1@model$validation_metrics
            model1@model$training_metrics@metrics$model_category
            h2o.confusionMatrix(model1)
            #test on testdata
            cat("Performance on test data:")
            perf<-h2o.performance(model1,test.h2o)
            perf
            #compute accuraccy on validation
            valResult <- h2o.predict(model1, validation.h2o,y=y.dep)
            predictions<-as.data.frame(valResult[,1])
            trueLabels<-c.validation$V65
            correct<-0
            for(i in 1:dim(predictions)[1]){
              if(as.numeric(predictions$predict[i])==as.numeric(c.validation$V65[i])){
                correct<-correct+1
              }
            }
            acc_V<-correct/dim(predictions)[1]
            cat("Accuraccy on validation set:",acc_V)
            #compute accuraccy on test
            testResult <- h2o.predict(model1, test.h2o,y=y.dep)
            predictions<-as.data.frame(testResult[,1])
            trueLabels<-test$V65
            correct<-0
            for(i in 1:dim(predictions)[1]){
              if(as.numeric(predictions$predict[i])==as.numeric(test$V65[i])){
                correct<-correct+1
              }
            }
            acc<-correct/dim(predictions)[1]
            cat("Accuraccy on test set:",acc)
            resultsTab[nrow(resultsTab)+1,] <- c(errF,hL,hU,lR,mS,iS,acc_V,acc,as.numeric(d)[3]/60)
            
            if(acc>bestAcc){
              bestModel<-model1
              bestAcc<-acc
            }
          }
        }
      }
    }
  }
}

print("Highest accuracy model:")
resultsTab[which.max(resultsTab$Acc_test),]
print ("Best model summary and confusion matrix for training set")
h2o.confusionMatrix(bestModel)
print ("confusion matrix for test set with best model")
h2o.confusionMatrix(bestModel,test.h2o)

#for plotting
resultsTab$ErrorFunction<-factor(resultsTab$ErrorFunction, levels=errorFunc)
par(mfrow=c(2,2))
boxplot(as.numeric(resultsTab$Acc_test)~as.numeric(resultsTab$layers),data=resultsTab, main="Acc vs Layers", ylab="Accuraccy", xlab="Num layers")
boxplot(as.numeric(resultsTab$Acc_test)~as.numeric(resultsTab$hiddenUnits),data=resultsTab, main="Acc vs Units", xlab="Hidden units", ylab="Accuraccy")
boxplot(as.numeric(resultsTab$Acc_test)~as.numeric(resultsTab$ErrorFunction),data=resultsTab, main="Acc vs Error func", xlab="Error fun (1. Sum of Sq,2. CrossEntropy)", ylab="Accuraccy")
boxplot(as.numeric(resultsTab$Acc_test)~as.numeric(resultsTab$learnRate),data=resultsTab, main="Acc vs Rate", ylab="Accuraccy", xlab="Learning rate")
par(mfrow=c(1,1))

#EXP 1b
##ALL data is already loaded

######Set model hyper-parameters
#hiddenLayers<-c(1,2,3)
#hiddenUnits<-c(100,200,300)
#learningRates<-c(0.005,0.01)
#momentumStart<-c(0,0.5)
#inputScaling<-c(T,F)
#errorFunc<-c("CrossEntropy")
act<-c("Rectifier","Tanh")
hiddenLayers<-c(1)
hiddenUnits<-c(100,200)
learningRates<-c(0.01)
momentumStart<-c(0)
inputScaling<-c(T)
errorFunc<-c("Quadratic")
##Table to write results
header2<-c("activation","layers","hiddenUnits","learnRate","momentumStart","Scale","Acc_val","Acc_test","Time in min")
resultsTab2<-data.frame(matrix(ncol = 9, nrow = 0))
colnames(resultsTab2)<-header2
#set seed for reproducible results
set.seed(1654)
#save the best model i.e. highest accuraccy on test set
bestModelb<-NULL
bestAcc<-0
#make all comniations of the parameters
for(a in act){
  for(hL in hiddenLayers){
    for(hU in hiddenUnits){
      for(lR in learningRates){
        for(mS in momentumStart){
          for(iS in inputScaling){
            
            #build model and calculate accuraccy
            s <- proc.time() #start time
            model1<- h2o.deeplearning(x=predictors,y = y.dep,training_frame = train.h2o, validation_frame=validation.h2o,hidden = c(rep(hU),hL), activation = a ,epochs = 100,loss=errorFunc,rate=lR,momentum_start = mS,standardize = iS,adaptive_rate=F)
            d <- proc.time()  - s #end time
            
            #test on testdata
            cat("Performance on test data:")
            perf<-h2o.performance(model1,test.h2o)
            perf
            #compute accuraccy on validation
            valResult <- h2o.predict(model1, validation.h2o,y=y.dep)
            predictions<-as.data.frame(valResult[,1])
            trueLabels<-c.validation$V65
            correct<-0
            for(i in 1:dim(predictions)[1]){
              if(as.numeric(predictions$predict[i])==as.numeric(c.validation$V65[i])){
                correct<-correct+1
              }
            }
            acc_V<-format(correct/dim(predictions)[1],digits = 4)
            cat("Accuraccy on validation set:",acc_V)
            #compute accuraccy on test
            testResult <- h2o.predict(model1, test.h2o,y=y.dep)
            predictions<-as.data.frame(testResult[,1])
            trueLabels<-test$V65
            correct<-0
            for(i in 1:dim(predictions)[1]){
              if(as.numeric(predictions$predict[i])==as.numeric(test$V65[i])){
                correct<-correct+1
              }
            }
            acc<-format(correct/dim(predictions)[1],digits = 4)
            cat("Accuraccy on test set:",acc)
            resultsTab2[nrow(resultsTab2)+1,] <- c(a,hL,hU,lR,mS,iS,acc_V,acc,format(as.numeric(d)[3]/60,digits = 2))
            
            if(acc>bestAcc){
              bestModelb<-model1
              bestAcc<-acc
            }
          }
        }
      }
    }
  }
}

#for plotting
resultsTab2$activation<-factor(resultsTab2$activation, levels=act)
par(mfrow=c(2,2))
boxplot(as.numeric(resultsTab2$Acc_test)~as.numeric(resultsTab2$layers),data=resultsTab2, main="Acc vs Layers", ylab="Accuraccy", xlab="Num layers")
boxplot(as.numeric(resultsTab2$Acc_test)~as.numeric(resultsTab2$hiddenUnits),data=resultsTab2, main="Acc vs Units", xlab="Hidden units", ylab="Accuraccy")
boxplot(as.numeric(resultsTab2$Acc_test)~as.numeric(resultsTab2$activation),data=resultsTab2, main="Acc vs Error func", xlab="Activation fun (1. ReLU,2. tanh)", ylab="Accuraccy")
boxplot(as.numeric(resultsTab2$Acc_test)~as.numeric(resultsTab2$learnRate),data=resultsTab2, main="Acc vs Rate", ylab="Accuraccy", xlab="Learning rate")
par(mfrow=c(1,1))


#Sol2 Lenet

h2o.shutdown(prompt=FALSE)


#sol2
# Load MXNet
require(mxnet)
#set train and test data
train_cn <- data.matrix(train)
train_x <- t(train_cn[,1:64])
train_y <- train_cn[,65]
train_array <- train_x

test_cn<- data.matrix(test)
test_x <- t(test_cn[,1:64])
test_y <- test[,65]
test_array <- test_x
##missed steps
#resize to 8x8 image
dim(train_array) <- c(8, 8, 1, ncol(train_x))
dim(test_array) <- c(8, 8, 1, ncol(test_x))

data <- mx.symbol.Variable('data')

#define hyperparameter space
#K<-c(1,2)
#numF<-c(1,2)
#hid1<-c(100,200,500)
#hid2<-c(2,10,50)
#learnRate<-c(0.005,0.01)
##test
K<-c(2)
numF<-c(20)
hid1<-c(500)
hid2<-c(40)
learnRate<-c(0.1)

#create table for results
##Table to write results
header3<-c("Num.Convlayers","Units in conv1","Units in conv2","kernel","NumFilter","Rate","Acc_train","Acc_test","Time in min")
resultsTab3<-data.frame(matrix(ncol = 9, nrow = 0))
colnames(resultsTab3)<-header3

bestModelc<-NULL
bestAcc<-0
#error func cross entropy, hidden activation ReLU
for (h1 in hid1) {
  for (h2 in hid2) {
    for (ks in K) {
      for (f in numF) {
        for (r in learnRate) {
          cat(h1,h2,ks,f,r)
          s <- proc.time() #start time
          # 1st convolutional layer
          conv_1 <-
            mx.symbol.Convolution(
              data = data,
              kernel = c(ks,ks),
              num_filter = f
            )
          relu_1 <-
            mx.symbol.Activation(data = conv_1, act_type = "relu")
          pool_1 <-
            mx.symbol.Pooling(
              data = relu_1,
              pool_type = "max",
              kernel = c(ks,ks)
            )
          # 2nd convolutional layer
          conv_2 <-
            mx.symbol.Convolution(
              data = pool_1,
              kernel = c(ks,ks),
              num_filter = f
            )
          relu_2 <-
            mx.symbol.Activation(data = conv_2, act_type = "relu")
          pool_2 <-
            mx.symbol.Pooling(
              data = relu_2,
              pool_type = "max",
              kernel = c(ks,ks)
            )
          # 1st fully connected layer
          flat <- mx.symbol.Flatten(data = pool_2)
          fcl_1 <-
            mx.symbol.FullyConnected(data = flat, num_hidden = h1)
          relu_3 <-
            mx.symbol.Activation(data = fcl_1, act_type = "relu")
          # 2nd fully connected layer
          fcl_2 <-
            mx.symbol.FullyConnected(data = relu_3, num_hidden = h2)
          # Output
          NN_model <-
            mx.symbol.SoftmaxOutput(data = fcl_2, name = 'softmax')
          # Set seed for reproducibility
          mx.set.seed(100)
          #use CPU
          device <- mx.cpu()
          # Train whole training data
          model <- mx.model.FeedForward.create(
            NN_model,
            X = train_array,
            y = train_y,
            ctx = device,
            num.round = 10,
            array.batch.size = 100,
            learning.rate = r,
            eval.metric = mx.metric.accuracy,
            epoch.end.callback = mx.callback.log.train.metric(100),
            verbose = T
          )
          d <- proc.time()  - s #end time
          cat("Time to build",format(as.numeric(d)[3]/60,digits = 2),"Mins")
          #accuraccy on train set
          predict_probs <- predict(model, train_array)
          predicted_labels <- max.col(t(predict_probs)) - 1
          correct <- 0
          for (i in 1:length(predicted_labels)) {
            if (as.numeric(predicted_labels[i]) == as.numeric(train$V65[i])) {
              correct <- correct + 1
            }
          }
          acc_tr <- format(correct / length(predicted_labels), digits = 4)
          cat("ConvNet Accuraccy on train set:", acc)
          #accuraccy on test set
          predict_probs <- predict(model, test_array)
          predicted_labels <- max.col(t(predict_probs)) - 1
          correct <- 0
          for (i in 1:length(predicted_labels)) {
            if (as.numeric(predicted_labels[i]) == as.numeric(test$V65[i])) {
              correct <- correct + 1
            }
          }
          acc <- format(correct / length(predicted_labels), digits = 4)
          cat("ConvNet Accuraccy on test set:", acc)
          
          #add to table
          resultsTab3[nrow(resultsTab3)+1,] <- c(2,h1,h2,paste("(",ks,",",ks,")",sep=""),f,r,acc_tr,acc,format(as.numeric(d)[3]/60,digits = 2))
          
          #choose best model
          if(acc>bestAcc){
            bestModelc<-model
            bestAcc<-acc
          }
          
        }
      }
    }
  }
}

confusion_matrix <- table(predicted_labels, t(test_y))

