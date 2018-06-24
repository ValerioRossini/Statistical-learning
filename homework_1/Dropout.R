load("supernova.RData")
library(glmnet)
X=as.matrix(supernova[,c(-1,-12)])
y=supernova$Magnitude

ll=glmnet(X,y)
fit.cv=cv.glmnet(X,y)
yhat=predict(fit.cv,X)
mean( (yhat - y)^2)

dropout.noise=function(X,delta){
  n=dim(X)[1]
  k=dim(X)[2]
  X.tilde=matrix(NA,n,k)
  for (i in 1:n){
    for (j in 1:k){
      eps=rbinom(1,1,delta)*(1/(1-delta))
      X.tilde[i,j]=X[i,j]*eps
    }
  }
   return(X.tilde)
}

M=100
n=dim(X)[1]
mse.delta=rep(NA,10)

d.grid=seq(0.1,0.9,length.out = 10)
for (idx in 1:length(d.grid)){
  d=d.grid[idx]
  mse.vec=rep(NA,M)
  X.t1=0
    for (m in 1:M){
      X.t=dropout.noise(X,d)
      X.t1=X.t1+X.t
    }
    X.finale=X.t1/M
    lm1=lm(y ~ X.finale)
    y.dropout=predict(lm1,data.frame(X.finale))
    mse.delta[idx]=(mean(y.dropout - y)^2)
}

mse.delta
