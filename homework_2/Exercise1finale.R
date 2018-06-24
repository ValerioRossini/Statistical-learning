# load the necessary packages
library(caret)
library(MASS)
library(h2o)
library(randomForest)
library(xgboost)
library(data.table)
library(mlr)

# load the file RData that contains in the first column the dependent variable, and in the other columns the features
load("featuresFinal.RData")

# divide the dataframe into train and test
idx.tr <- createDataPartition(y = df$y, p = .65, list = FALSE)
X.tr <- as.matrix(df[ idx.tr,-1])
X.te <- as.matrix(df[-idx.tr,-1])
y.tr <- df[ idx.tr,1]
y.te <- df[-idx.tr,1]

train=data.frame(y=y.tr,X.tr)
test=data.frame(y=y.te,X.te)

# try different models in order to find the best accuracy. The models used in this exercise will be:
# LDA
# deep learn
# random forest
# xgboost

# create a vector that cointains the accuracy for each model 
accuracy.vec <- c()

# LDA ---------------------------------------------------------------------

# Fit the LDA model
mod.lda = lda(y ~ ., data = train)
class(mod.lda)
names(mod.lda)

# Prediction error on test
pred.lda = predict(mod.lda, data.frame(X.te))
names(pred.lda)

# Take a look
head(pred.lda$class)        # predicted class
head(pred.lda$posterior)    # per class posterior prob

# Missclassification error on test
mean(pred.lda$class != test$y)*100
confusionMatrix(pred.lda$class,test$y)
lda.accuracy <- confusionMatrix(pred.lda$class,test$y)$overall[1]
lda.accuracy
accuracy.vec[1] <- lda.accuracy

# h20--------------

#start h2o
localH2o <- h2o.init(nthreads = -1, max_mem_size = "20G")

#load data on H2o
trainh2o <- as.h2o(train)
testh2o <- as.h2o(test)

#set variables
y <- "y"
x <- setdiff(colnames(trainh2o),y)



# Hyper-parameter Tuning with Grid Search
#set parameter space
activation_opt <- c("Rectifier","RectifierWithDropout", "Maxout","MaxoutWithDropout")
hidden_opt <- list(c(200,200),c(20,15),c(50,50,50))
l1_opt <- c(0,1e-3,1e-5)
l2_opt <- c(0,1e-3,1e-5)

hyper_params <- list( activation=activation_opt,
                      hidden=hidden_opt,
                      l1=l1_opt,
                      l2=l2_opt )

#set search criteria
search_criteria <- list(strategy = "RandomDiscrete", max_models=10)

library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

#train model
dl_grid <- h2o.grid("deeplearning"
                    ,grid_id = "deep_learn"
                    ,hyper_params = hyper_params
                    ,search_criteria = search_criteria
                    ,training_frame = trainh2o
                    ,x=x
                    ,y=y
                    ,nfolds = 4
                    ,epochs = 100)

#get best model
d_grid <- h2o.getGrid("deep_learn",sort_by = "accuracy") #MaxoutWithDropout c(50,50,50) 1e-3 1e-3
best_dl_model <- h2o.getModel(d_grid@model_ids[[10]])
h2o.performance (best_dl_model,xval = T)
pred <- h2o.predict(best_dl_model, testh2o)
pred
test$Accuracy <- pred$predict == testh2o$y
deep.learn.accuracy <- mean(test$Accuracy)   
deep.learn.accuracy
accuracy.vec[2] <- deep.learn.accuracy

#compute variable importance and performance
h2o.varimp_plot(best_dl_model,num_of_features = 20)
h2o.performance(best_dl_model,xval = T)
h20.plot <- h2o.varimp_plot(best_dl_model,num_of_features = 20)

# RandoForest------------

n <- names(train)
f <- as.formula(paste("y  ~", paste(n[2:length(n)], collapse = " + ")))
f

# fit the model
mod.rf = randomForest(f, data=train
                        ,ntree=300
                        ,mtry=30
                        ,sampsize=nrow(train)
                        ,importance=TRUE)

plot(mod.rf)
print(mod.rf)

# compute the variable importance using before the MeanDecreaseAccuracy and then MeanDecreaseGini
rf_plot <- varImpPlot(mod.rf)
rf_plot_MeanDecreaseAccuracy <- rf_plot[order(rf_plot[,1],decreasing = T),]
row.names(rf_plot_MeanDecreaseAccuracy)
rf_plot_MeanDecreaseGini <- rf_plot[order(rf_plot[,2],decreasing = T),]
row.names(rf_plot_MeanDecreaseGini)
pred.rf <- predict(mod.rf, data.frame(X.te), type="response")   
confusionMatrix(pred.rf,test$y)
rf.accuracy <- confusionMatrix(pred.rf,test$y)$overall[1]
rf.accuracy
accuracy.vec[3] <- rf.accuracy

# XGboost-------------

#convert data frame to data table
setDT(train) 
setDT(test)

# some trasformation
labels <- train$y 
ts_label <- test$y
new_tr <- as.matrix(train[,-1])
new_ts <- as.matrix(test[,-1])

#convert factor to numeric 
labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1

#preparing matrix 
dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)

#default parameters. the objective "multi:softmax" is used for the classification
params <- list(booster = "gbtree", 
               objective = "multi:softmax",
               num_class=5, 
               eta=0.3, 
               gamma=0, 
               max_depth=6, 
               min_child_weight=1, 
               subsample=1, 
               colsample_bytree=1)

# compute the cross validation and then find the best iteration
xgbcv <- xgb.cv( params = params, 
                 data = dtrain, 
                 nrounds = 200, 
                 nfold = 10, 
                 showsd = T, 
                 stratified = T, 
                 print.every_n = 10, 
                 early_stopping_rounds = 20, 
                 maximize = F,
                 metrics="merror")


nrou=xgbcv$best_iteration
#first default - model training
xgb1 <- xgb.train (params = params, 
                   data = dtrain, 
                   nrounds = nrou, 
                   watchlist = list(val=dtest,train=dtrain), 
                   print.every_n = 10, 
                   early_stopping_rounds = 10, 
                   maximize = F , 
                   eval_metric = "mlogloss")



#model prediction
xgbpred <- predict (xgb1,dtest)
confusionMatrix (xgbpred, ts_label)

#view variable importance plot
mat <- xgb.importance (feature_names = colnames(new_tr),model = xgb1)
xgb.plot.importance (importance_matrix = mat[1:20])

# Let's try to improve

#create tasks
traintask <- makeClassifTask (data = train,target = "y")
testtask <- makeClassifTask (data = test,target = "y")

#do one hot encoding`<br/> 
#traintask <- createDummyFeatures (obj = traintask,target = "y") 
#testtask <- createDummyFeatures (obj = testtask,target = "target")

#create learner
lrn <- makeLearner("classif.xgboost",
                   predict.type = "response")

lrn$par.vals <- list( objective = "multi:softprob",
                      num_class=5, 
                      eval_metric="mlogloss", 
                      nrounds=100L, 
                      eta=0.1)

#set parameter space
params <- makeParamSet( makeDiscreteParam("booster",
                                          values = c("gbtree")), 
                        makeIntegerParam("max_depth",
                                         lower = 3L,
                                         upper = 10L), 
                        makeNumericParam("min_child_weight",
                                         lower = 1L,
                                         upper = 10L), 
                        makeNumericParam("gamma",
                                         lower = 1L,
                                         upper = 5L),
                        makeNumericParam("subsample",
                                         lower = 0.5,
                                         upper = 1), 
                        makeNumericParam("colsample_bytree",
                                         lower = 0.5,
                                         upper = 1),
                        makeNumericParam("lambda",
                                         lower = 0,
                                         upper = 1),
                        makeNumericParam("alpha",
                                         lower = 0,
                                         upper = 1))

#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

#search strategy
ctrl <- makeTuneControlRandom(maxit = 15L)

#set parallel backend
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

#parameter tuning
mytune <- tuneParams(learner = lrn, 
                     task = traintask, 
                     resampling = rdesc, 
                     measures = acc, 
                     par.set = params, 
                     control = ctrl, 
                     show.info = T)
mytune$y

#set hyperparameters
lrn_tune <- setHyperPars(lrn,
                         par.vals = mytune$x)

#train model
xgmodel <- train(learner = lrn_tune,
                 task = traintask)

#predict model
xgpred <- predict(xgmodel,testtask)
confusionMatrix(xgpred$data$response,xgpred$data$truth)
xgb.accuracy <- confusionMatrix(xgpred$data$response,xgpred$data$truth)$overall[1]
xgb.accuracy
accuracy.vec[4] <- xgb.accuracy

cbind(lda.accuracy=accuracy.vec[1], deep.learn.accuracy=accuracy.vec[2],
      rf.accuracy=accuracy.vec[3], xgb.accuracy=accuracy.vec[4])
