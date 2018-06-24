load('tempo_hw.RData')
# Data
X=tempo_hwdata[,6:101]
predX=tempo_hwdata[,2:5]

X2=X[which(colSums(sapply(X, is.na))<6)]
which(colSums(sapply(X2, is.na))<6)

y=tempo_hwdata$tapping

#Evaluation part ----------
evaluation = function (pred,y){
  res=0
  for(i in c(1/2,1/3,1,2,3))
  {
    if(pred<y*1.04*i & pred> y*0.96*i){
      res=1
    }
  }
  return(res)
}
evaluation = Vectorize(evaluation)

# treatment na--------
#library(mice)
#Xsinna=mice(X2,m=5,maxit=20,method='pmm',seed=500)
load("Xsinna.RData")

# LASSO CV----
require(glmnet)
library(modelr)
Xlas=complete(Xsinna,2)
Xlas=as.matrix(Xlas)
tempo=cbind(y,Xlas)

err.train.cp=matrix(NA,nrow=5,ncol=16)
err.test.cp=matrix(NA,nrow=5,ncol=4)
folds <- crossv_kfold(data.frame(tempo), k = 5)
folds
for (i in 1:5){

  X.train <- Xlas[ folds$train[[i]]$idx, ]
  y.train <- y [folds$train[[i]]$idx]
  X.test  <- Xlas[folds$test[[i]]$idx, ]
  y.test <- y [folds$test[[i]]$idx]
  
    # LASSO/Cp----
  # For each fit0$lambda, we need to built its Cp risk estimate
  fit0=glmnet(X.train,y.train)
  #length(fit0$lambda)
  n=nrow(X.train)
  yhatcp = predict(fit0,X.train)
  #dim(yhatcp) # one col for each lambda
  
  RSS = colSums((y.train- yhatcp)^2 )
  Rhat=(1/n)*RSS
  #plot(log(fit0$lambda),Rhat)
  s2=RSS/(n-fit0$df -1)
  Cp=Rhat +(2*s2*fit0$df)/n
  #plot(log(fit0$lambda),Cp,type='l')
  lam.cp=fit0$lambda[which.min(Cp)]
  #plot(fit0,xvar='lambda',label=T)
  #abline(v=log(lam.cp),lty=3)
  
  # prediction with lambda.cp
  fit2=glmnet(X.train,y.train,lambda=lam.cp)
  yhat2=predict(fit2,X.train)
  #plot(yhat2, y.train,pch=21,bg='purple')
  #mean( (y.train - yhat2)^2)
  err.train.cp[i,]=evaluation( yhat2 ,y.train)
  
  #test part
  #mean( (predict(fit2,X.test) - y.test )^2)
  err.test.cp[i,]=evaluation( predict(fit2,X.test) ,y.test)
}

rowSums(err.train.cp)
rowSums(err.test.cp)
