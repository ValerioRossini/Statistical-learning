load('tempo_hw.RData')
# Data
X=tempo_hwdata[,6:101]
predX=tempo_hwdata[,2:5]

X2=X[which(colSums(sapply(X, is.na))<6)]
which(colSums(sapply(X2, is.na))<6)

y=tempo_hwdata$tapping

tempo=cbind(y,X2)
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
library(mice)
Missing_indices = sapply(X,function(x)sum(is.na(x)))
Missing_Summary = data.frame(index = names(X),Missing_Values=Missing_indices)
Missing_Summary[Missing_Summary$Missing_Values > 0,]

#Xsinna=mice(X2,m=5,maxit=20,method='pmm',seed=500)
load("Xsinna.RData")

# LASSO ----
require(glmnet)
Xlas=complete(Xsinna,2)
Xlas=as.matrix(Xlas)

# Data-split
n=16
trset      <- sample(1:20,n,replace = F)
X.train <- Xlas[ trset, ]
y.train <- y [trset]
X.test  <- Xlas[-trset, ]
y.test <- y [-trset]

# How to choose lambda?? Pick lambda by CV
fit.cv=cv.glmnet(X.train,y.train)
plot(fit.cv)
lam.cv=fit.cv$lambda.min

# prediction with lambda.cv
fit1=glmnet(X.train,y.train,lambda=lam.cv)
yhat1=predict(fit1,X.train)
plot(yhat1, y.train,pch=21,bg='purple')
mean( (y.train - yhat1)^2)
evaluation( yhat1 ,y.train)

#test part
mean( (predict(fit1,X.test) - y.test )^2)
evaluation( predict(fit1,X.test) ,y.test)

# LASSO/Cp----
# For each fit0$lambda, we need to built its Cp risk estimate
fit0=glmnet(X.train,y.train)
length(fit0$lambda)
n=nrow(X.train)
yhatcp = predict(fit0,X.train)
dim(yhatcp) # one col for each lambda

RSS = colSums((y.train- yhatcp)^2 )
Rhat=(1/n)*RSS
plot(log(fit0$lambda),Rhat)
s2=RSS/(n-fit0$df -1)
Cp=Rhat +(2*s2*fit0$df)/n
plot(log(fit0$lambda),Cp,type='l')
lam.cp=fit0$lambda[which.min(Cp)]
plot(fit0,xvar='lambda',label=T)
abline(v=log(lam.cp),lty=3)

# prediction with lambda.cp
fit2=glmnet(X.train,y.train,lambda=lam.cp)
yhat2=predict(fit2,X.train)
plot(yhat2, y.train,pch=21,bg='purple')
mean( (y.train - yhat2)^2)
evaluation( yhat2 ,y.train)

#test part
mean( (predict(fit2,X.test) - y.test )^2)
evaluation( predict(fit2,X.test) ,y.test)


pesi=1/apply(tempo_hwdata[,2:5],1,FUN = var)
evaluation(tempo_hwdata[,5],y)
round(pesi,4)

library(MASS)
x=complete(Xsla,2)
tempo=cbind(y,x)
mod.all=lm(y~.,data=x)
mod.all 
Step3 <- stepAIC(mod.all, direction="forward")




