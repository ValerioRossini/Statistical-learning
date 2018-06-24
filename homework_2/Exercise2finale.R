library(splines)
library(sm)
library(mgcv)

# load the file "ore.RData
df <- load("ore.RData")

# take a look 
ore
names(ore)
head(ore)
str(ore)
attributes(ore)


rss.fun=function(y,F.hat,alpha){
  out=sum( (y-alpha-rowSums(F.hat))^2)
  return(out)
}

# create a function that will return the smooth functions of all variables
backfitting.GCV=function(y,X,tol=1e-6,maxiter=100){
  n=dim(X)[1]
  p=dim(X)[2]
  # set alpha.hat as the mean of y
  alpha.hat=mean(y)
  # create a matrix 0x0 that will contain successfully the generic vector of in-sample predictions based on the jth covariate 
  F.hat=matrix(0,n,p)
  # compute residual sum of squares of the initial estimate
  rss0=rss.fun(y,F.hat,alpha.hat)
  it=0
  # we use a while to repeat the backfitting loop until the rss of the current estimate doesn't change enough
  while(it < maxiter){
    rss1=rss0
    it=it+1
    for (j in 1:p){
      Fk.hat = as.matrix(F.hat[,-j])
      # calculate partial residuals
      rj = ( y-alpha.hat- rowSums(Fk.hat))
      # set fj equal to the in-sample predictions obtained by smoothing the residuals rj with respect to xj
      fj.spl =  smooth.spline( X[,j],rj)
      fj=predict(fj.spl, X[,j])
      # recenter by subtracting the mean
      F.hat[,j] = fj$y-mean(fj$y)
    }
    rss0=rss.fun(y,F.hat,alpha.hat)
    e=abs(rss0-rss1)
    if(e<tol*rss0){
      return(F.hat)
    }
  }
  return(F.hat)
}

# we do the same procedures, but in this case in smooth.spline we set "cv" as TRUE in order to compute the LOO cv
# create a function that will return the smooth functions of all variables
backfitting.LOO=function(y,X,tol=1e-6,maxiter=100){
  n=dim(X)[1]
  p=dim(X)[2]
  # set alpha.hat as the mean of y
  alpha.hat=mean(y)
  # create a matrix 0x0 that will contain successfully the generic vector of in-sample predictions based on the jth covariate
  F.hat=matrix(0,n,p)
  # compute residual sum of squares of the initial estimate
  rss0=rss.fun(y,F.hat,alpha.hat)
  it=0
  # we use a while to repeat the backfitting loop until the rss of the current estimate doesn't change enough
  while(it < maxiter){
    rss1=rss0
    it=it+1
    for (j in 1:p){
      Fk.hat = as.matrix(F.hat[,-j])
      # calculate partial residuals
      rj = ( y-alpha.hat- rowSums(Fk.hat))
      # set fj equal to the in-sample predictions obtained by smoothing the residuals rj with respect to xj
      fj.spl =  smooth.spline( X[,j],rj, cv =T) 
      fj=predict(fj.spl, X[,j])
      # recenter by subtracting the mean
      F.hat[,j] = fj$y-mean(fj$y)
    }
    rss0=rss.fun(y,F.hat,alpha.hat)
    e=abs(rss0-rss1)
    if(e<tol*rss0){
      return(F.hat)
    }
  }
  return(F.hat)
}

# choose the width as dependent variable
y=ore$width
# using the remain variable as independent variables 
X=ore[,-3]
# using the function backfitting.GCV to compute the F hat matrix 
F.hat=backfitting.GCV(y,X)
# using the function backfitting.LOO to compute the F hat matrix
F.hat.cv=backfitting.LOO(y,X)

alpha.hat=mean(y)
# fit the model 
fit0=alpha.hat+rowSums(F.hat)
fit0.cv=alpha.hat+rowSums(F.hat.cv)

rss.fun(y,F.hat,alpha.hat)
sum((y-fit0)^2)
sum((y-fit0.cv)^2)

par(mfrow=c(3,2))
plot(X[,1],F.hat[,1],ylim=c(-8,4),xlim=c(-10,80),main="Variable t1 with gcv")
plot(X[,2],F.hat[,2],ylim=c(-8,4),xlim=c(-80,10),main="Variable t2 with gcv")

plot(X[,1],F.hat.cv[,1],ylim=c(-8,4),xlim=c(-10,80),main="Variable t1 with LOO")
plot(X[,2],F.hat.cv[,2],ylim=c(-8,4),xlim=c(-80,10),main="Variable t2 with LOO")



# use the package mgcv
ore.gam <- gam(width ~ s(t1) + s(t2), data = ore)
sum((y-ore.gam$fitted.values)^2)
plot(ore.gam,main = c("Variable estimated by GAM"))

#par(mfrow=c(1,1))

par(mfrow=c(1,2))
# plot the difference between the true values and the predicted values using the package mgcv and gam
plot(y-ore.gam$fitted.values)
abline(h=0,col="red")
plot(y-fit0)
abline(h=0,col="red")


plot(y,ore.gam$fitted.values)
plot(y,fit0)
