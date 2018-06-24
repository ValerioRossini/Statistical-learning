

# function creation -------------------------------------------------------

library(splines)
library(sm)
library(caret)


rss.fun=function(y,F.hat,alpha){
  out=sum( (y-alpha-rowSums(F.hat))^2)
  return(out)
}

backfitting.cv=function(y,X,tol=1e-3,maxiter=1e5, lambda= 0.1, df=20){
  n=dim(X)[1]
  p=dim(X)[2]
  nu.j <- rep(NA,p)
  alpha.hat=mean(y)
  F.hat=matrix(0,n,p)
  # Residual sum of squares of the initial estimate
  rss0=rss.fun(y,F.hat,alpha.hat)
  #print(rss0)
  mod <- list()
  features.active <- rep(0,p)
  it=0
  while(it < maxiter){
    rss1=rss0
    it=it+1
    #print(it)
    #print(rss1)
    for (j in 1:p){
      Fk.hat = as.matrix(F.hat[,-j])
      rj = ( y-  alpha.hat- rowSums(Fk.hat))
      fj.spl =  smooth.spline(x = X[,j],rj, df=df)
      mod[[j]] <- fj.spl
      uj=predict(fj.spl, X[,j])
      l.hat.2 <- mean((uj$y)^2)
      features.active[j] <- max(0, (1- (lambda/sqrt(l.hat.2))))
      fj<- features.active[j]*uj$y
      F.hat[,j] = fj-mean(fj)
      nu.j[j] = fj.spl$df*(norm(F.hat[,j],type = '2')!=0)
      #print(j)
    }
    rss0=rss.fun(y,F.hat,alpha.hat)
    e=abs(rss0-rss1)
    #print(e)
    if(e<tol*rss0){
      df.lam <- sum(nu.j)
      gcv.lam <- mean((y- alpha.hat - rowSums(F.hat))^2)/(1 - df.lam/n)^2
      return(list(values =F.hat, gcv=gcv.lam, model = mod, activation = features.active))
    }
  }
  df.lam <- sum(nu.j)
  gcv.lam <- mean((y- alpha.hat - rowSums(F.hat))^2)/(1 - df.lam/n)^2
  return(list(values =F.hat, gcv=gcv.lam, model = mod, activation = features.active))
}

prediction.data <- function(X.te, models.pre, activation){
  n.new=dim(X.te)[1]
  p.new=dim(X.te)[2]
  F.new= matrix(nrow = n.new, ncol = p.new)
  for(el in 1:p.new){
    F.new[,el]= c(predict(models.pre[[el]] , x =X.te[,el])$y)
    F.new[,el] <- activation[el]*F.new[,el]
  }
  return(F.new)
}

backfitting.tuning.lambda <- function(y,X, n=10, df= df){
  res <- matrix(0, nrow = n, ncol = 2)
  it <- 1
  for(lam in seq(0,0.1,length.out = n)){
    print(lam)
    back <- backfitting.cv(y,X,tol=1e-3,maxiter=100, lambda= lam, df = df)
    res[it,] <- c(lam, back$gcv)
    it <- it + 1
  }
  return(res)
}

backfitting.kfold <- function(X,y, K_fold, lambda, df = df){
  folds <- createFolds(y,k= K_fold)
  mse <- c()
  for(i in 1:K_fold){
    X.train <- X[folds[[i]],]
    X.test <- X[-folds[[i]],]
    y.train <- y[folds[[i]]]
    y.test <- y[-folds[[i]]]
    backfit.cv <- backfitting.cv(y.train, X.train,lambda = lambda, df = df)
    pred <- prediction.data(X.te, backfit.cv$model, backfit.cv$activation)
    predic <- alpha.hat + rowSums(pred)
    mse[i] <- mean((y.te - predic)^2)
  }
  return(mean(mse))
}



# Generating data ---------------------------------------------------------

# Generating training data
n = 150; d=200
X.tr <- 0.5*matrix(runif(n*d),n,d) + matrix(rep(0.5*runif(n),d),n,d)

# Generating response
y.tr <- -2*sin(5*X.tr[,1]) + 5*X.tr[,2]^2 -(1/3) + 6*X.tr[,3]- (1/2) + exp(-2*X.tr[,4]) + exp(-1) - 1


# Generating testing data
n = 500; d=200
X.te <- 0.5*matrix(runif(n*d),n,d) + matrix(rep(0.5*runif(n),d),n,d)

# Generating response
y.te <- -2*sin(5*X.te[,1]) + 5*X.te[,2]^2 -(1/3) + 6*X.te[,3]- (1/2) + exp(-2*X.te[,4]) + exp(-1) - 1



# Model selection ----------------------------------------------------------------

# Training
backfit <- backfitting.cv(y.tr, X.tr,lambda = 0.01)

# GCV to choose lambda
table <- backfitting.tuning.lambda(y.tr,X.tr, df = 18)
#save(table, file = 'table.RData')
load('table.RData')
plot(table[[29]], type='l')

# Cross Validation 5-fold for lambda and degrees of freedom for smooth.spline

rdf = 5
n=5
value <- matrix(nrow = n*rdf, ncol = 3)
it=1
for(df in seq(16,25,length.out = rdf)){
  for(l in seq(0.1,2,length.out = n)){
    value[it,] <- c(l,backfitting.kfold(X.tr,y.tr,K_fold = 5,l, df=df), df)
    print(it)
    it=it+1
  }
}
#save(value, file = 'cross_validation.RData')
#load('cross_validation.RData')
plot(value[c(1,2,3,4,5),c(1,2)],type='l')
# best option df = 16; lambda = 0.6


# GCV to choose lambda and df

table <- list()
for(df in seq(10,25,length.out = rdf)){
  table[[it]] <- backfitting.tuning.lambda(y.tr,X.tr,df=df,n = 5)
  it <- it + 1
}
#save(table, file = 'table.RData')
#load('table.RData')
plot(table[[1]], type='l')


# model fitting -----------------------------------------------------------

# fitting model to training data
backfit <- backfitting.cv(y.tr, X.tr, lambda = 0.07, df=18)


# train scoring
alpha.hat=mean(y.tr)
fit0=alpha.hat+rowSums(backfit$value)


# Evaluation on training data
sum((y.tr - fit0)^2)
mean((y.tr - fit0)^2)
par(mfrow=c(1,1), mar=c(2.5,2.5,2.5,2.5))
plot(y.tr, fit0)
lines(x = seq(from = 0,to = 9,length.out = 10), y =seq(from = 0,to = 9,length.out = 10),
      type = 'l', col='red' )
which(colSums(backfit$values>0)>0)



# plotting results
par(mfrow=c(2,2), mar=c(2,2,1.5,1.5))
plot(X.tr[,1],backfit$value[,1],main="Variable t1",col="blue")
points(X.tr[,1],-2*sin(5*X.tr[,1]),ylim=c(-2,3),xlim=c(0,1),main="Variable t1")

plot(X.tr[,2],alpha.hat+backfit$value[,2],ylim=c(-2,5),xlim=c(0,1),col="blue",main="Variable t2")
points(X.tr[,2],5*X.tr[,2]^2 -(1/3),ylim=c(-2,5),xlim=c(0,1))

plot(X.tr[,3],alpha.hat+backfit$value[,3],ylim=c(-2,7),xlim=c(0,1),col="blue",main="Variable t3")
points(X.tr[,3],6*X.tr[,3]- (1/2),ylim=c(-2,7),xlim=c(0,1))

plot(X.tr[,4],backfit$value[,4],ylim=c(-2,2),xlim=c(0,1),col="blue",main="Variable t4")
points(X.tr[,4],exp(-2*X.tr[,4]) + exp(-1) - 1,ylim=c(-2,2),xlim=c(0,1))



# Prediction --------------------------------------------------------------

# prediction on new data
pred <- prediction.data(X.te, backfit$model, backfit$activation)
predic <- alpha.hat + rowSums(pred)


# Evaluation
which(colSums(pred>0)>0)
sum((y.te - predic)^2)
mean((y.te - predic)^2)
par(mfrow=c(1,1), mar=c(2.5,2.5,2.5,2.5))
plot(y.te, predic)
lines(x = seq(from = 0,to = 20,length.out = 10), y =seq(from = 0,to =20,length.out = 10),
      type = 'l', col='red' )


# plotting results
par(mfrow=c(2,2), mar=c(1,1,1,1))
plot(X.te[,1],pred[,1],main="Variable t1",col="blue")
points(X.te[,1],-2*sin(5*X.te[,1]),ylim=c(-2,3),xlim=c(0,1),main="Variable t1")

plot(X.te[,2],5*X.te[,2]^2 -(1/3),ylim=c(-2,5),xlim=c(0,1),main="Variable t2")
points(X.te[,2],alpha.hat+pred[,2],ylim=c(-2,5),xlim=c(0,1),col="blue")

plot(X.te[,3],6*X.te[,3]- (1/2),ylim=c(-2,7),xlim=c(0,1),main="Variable t3")
points(X.te[,3],alpha.hat+pred[,3],ylim=c(-2,7),xlim=c(0,1),col="blue")

plot(X.te[,4],exp(-2*X.te[,4]) + exp(-1) - 1,ylim=c(-2,2),xlim=c(0,1),main="Variable t4")
points(X.te[,4],pred[,4],ylim=c(-2,2),xlim=c(0,1),col="blue")


# We run the simulation many time and each time our model fit the right features
# with very little MSE and fitting quite well the functions.


# SAM packet comparison ---------------------------------------------------

library(SAM)

# Training
trn = samQL(X.tr,y.tr)
trn

# plotting solution path
par(mfrow=c(1,1))
plot(trn)


## predicting response
tst = predict(trn,X.te)

# Evaluation
sum((y.te - tst[[1]][,30])^2)
mean((y.te - tst[[1]][,30])^2)
par(mfrow=c(1,1), mar=c(2.5,2.5,2.5,2.5))
plot(y.te, tst[[1]][,30])
lines(x = seq(from = -10,to = 20,length.out = 10), y =seq(from =-10,to =20,length.out = 10),
      type = 'l', col='red' )

