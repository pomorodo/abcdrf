########################################################
############## Comparison of some ABC methods ##########
############## on a Gaussian toy example      ##########
########################################################

# Loading of usefull packages
install.packages(c("abcrf", "abc", "doParallel", "mvtnorm", "MCMCpack"))   
install.packages("spatstat")   
library(abcrf)
library(abc)
library(doParallel) # for parallel computing
library(mvtnorm) # for multivariate normal distribution
library(spatstat) # for weighted.quantile function
library(MCMCpack) # for the inverse gamma distribution

# Sample size of y

n <- 10    

# Inverse gamma parameters

alpha <- 4
beta <- 3

# Function to compute quantiles from
# student distribution

qnst <- function(p, deg, loca, scale) {
  return(loca + scale * qt(p, df = deg))
}

# Simulation of the ABC reference table

set.seed(1) # for reproducibility 

N <- 10000  # size of the training set
theta1.train <- rep(NA, N)
theta2.train <- 1 / rgamma(N, shape = alpha, rate = beta)
for (i in 1:N) {
  theta1.train[i] <- rnorm(1, 0, sqrt(theta2.train[i]))
}

y.ref <- matrix(NA, N, n)
for (i in 1:N) {
  y.ref[i, ] <- rnorm(n, theta1.train[i], sqrt(theta2.train[i]))
}

# Compute some summary statistics 

summa.train <- matrix(NA, N, 3)

for (i in 1:N) {
  summa.train[i, ] <- c(mean(y.ref[i, ]), var(y.ref[i, ]), mad(y.ref[i, ]))
}

ref.training <- cbind(theta1.train, theta2.train, summa.train)
colnames(ref.training) <- c("theta1", "theta2", "expectation", "variance", "mad")

# Simulation of the ABC test table

p <- 100    # size of the testing set
theta1.test <- rep(NA, p)

theta2.test <- 1 / rgamma(p, shape = alpha, rate = beta)
for (i in 1:p) {
  theta1.test[i] <- rnorm(1, 0, sqrt(theta2.test[i]))
}

y.test <- matrix(NA, p, n)

for (i in 1:p) {
  y.test[i, ] <- rnorm(n, theta1.test[i], sqrt(theta2.test[i]))
}

# Compute some summary statistics

summa.test <- matrix(NA, p, 3)

for (i in 1:p) {
  summa.test[i, ] <-
    c(mean(y.test[i, ]), var(y.test[i, ]), mad(y.test[i, ]))
}

ref.testing <- cbind(theta1.test, theta2.test, summa.test)
colnames(ref.testing) <- c("theta1", "theta2", "expectation", "variance", "mad")

# Compute the exact posterior expectations, variances and quantiles 
# for parameters theta1 and theta2

theta1.test.exact <- rep(NA, p)
theta2.test.exact <- rep(NA, p)
var1.test.exact <- rep(NA, p)
var2.test.exact <- rep(NA, p)
quant.theta1.test.freq <- matrix(NA, p, 2)
quant.theta2.test.freq <- matrix(NA, p, 2)

for (i in 1:p) {
  theta1.test.exact[i] <- sum(y.test[i, ]) / (n + 1)
  var1.test.exact[i] <-
    (beta + sum((y.test[i, ] - mean(y.test[i, ])) ^ 2)/2 + n*(mean(y.test[i, ]))^2 / (2*(n+1))  ) /
    ( (n + 1) * (alpha - 1 + n / 2) )
  theta2.test.exact[i] <-
    (beta + sum((y.test[i, ] - mean(y.test[i, ])) ^ 2)/2 + n*(mean(y.test[i, ]))^2 / (2*(n+1))  ) / (alpha - 1 + n / 2)
  var2.test.exact[i] <-
    (beta + sum((y.test[i, ] - mean(y.test[i, ])) ^ 2)/2 + n*(mean(y.test[i, ]))^2 / (2*(n+1))  ) ^ 2 / ((alpha - 1 + n / 2) ^ 2 * (alpha - 2 + n / 2))
  quant.theta1.test.freq[i, ] <-
    c(qnst(0.025, n + 2 * alpha, sum(y.test[i, ]) / (n + 1), sqrt(2 * (beta + sum((y.test[i, ] - mean(y.test[i, ])) ^ 2)/2 + n*(mean(y.test[i, ]))^2 / (2*(n+1))  ) / ((n + 1) * (n + 2 * alpha) ))),
      qnst(0.975, n + 2 * alpha, sum(y.test[i, ]) / (n + 1), sqrt(2 * (beta + sum((y.test[i, ] - mean(y.test[i, ])) ^ 2)/2 + n*(mean(y.test[i, ]))^2 / (2*(n+1))  ) / ((n + 1) * (n + 2 * alpha) ))))
  quant.theta2.test.freq[i, ] <- 
    c(1 / qgamma(0.975, shape = (n + 2 * alpha) / 2, rate = (beta + sum((y.test[i, ] - mean(y.test[i, ])) ^ 2)/2 + n*(mean(y.test[i, ]))^2 / (2*(n+1))  ) ),
      1 / qgamma(0.025, shape = (n + 2 * alpha) / 2, rate = (beta + sum((y.test[i, ] - mean(y.test[i, ])) ^ 2)/2 + n*(mean(y.test[i, ]))^2 / (2*(n+1))  )) )
}

# Add nNoise summary statistics simulated according 
# to a uniform(0,1) distribution

nNoise <- 50 # or 500

set.seed(3)  # for reproducibility 

summa.noise <- matrix(runif((N+p) * nNoise), N+p, nNoise) 
ref.training <- cbind(ref.training, summa.noise[1:N, ])
ref.testing <- cbind(ref.testing, summa.noise[(N+1):(N+p), ])

colnames(ref.training) <-
  c("theta1", "theta2", "expectation", "variance", "mad", c(1:nNoise))
colnames(ref.testing) <-
  c("theta1", "theta2", "expectation", "variance", "mad", c(1:nNoise))

# Add some others summary statistics 

y <- ref.training[, 1:2]
x <- ref.training[, -c(1:2)]

x <-cbind(x, x[, 1] + x[, 2], x[, 1] + x[, 3], x[, 2] + x[, 3], x[, 1] + x[, 2] +
            x[, 3], x[, 1] * x[, 2], x[, 1] * x[, 3], x[, 2] * x[, 3], x[, 1] * x[, 2] *
            x[, 3])
colnames(x) <-
  c("expectation",
    "variance",
    "mad",
    c(1:nNoise),
    "sum_esp_var",
    "sum_esp_mad" ,
    "sum_var_mad",
    "sum_esp_var_mad",
    "prod_esp_var",
    "prod_esp_mad",
    "prod_var_mad" ,
    "prod_esp_var_mad")

ytest <- ref.testing[, 1:2]
xtest <- ref.testing[, -c(1:2)]

xtest <-
  cbind(
    xtest,
    xtest[, 1] + xtest[, 2],
    xtest[, 1] + xtest[, 3],
    xtest[, 2] + xtest[, 3],
    xtest[, 1] + xtest[, 2] + xtest[, 3],
    xtest[, 1] * xtest[, 2],
    xtest[, 1] * xtest[, 3],
    xtest[, 2] * xtest[, 3],
    xtest[, 1] * xtest[, 2] * xtest[, 3]
  )
colnames(xtest) <-
  c("expectation",
    "variance",
    "mad",
    c(1:nNoise),
    "sum_esp_var",
    "sum_esp_mad" ,
    "sum_var_mad",
    "sum_esp_var_mad",
    "prod_esp_var",
    "prod_esp_mad",
    "prod_var_mad" ,
    "prod_esp_var_mad")

data.theta1 <- data.frame(theta1 = y[,1], x)
data.theta2 <- data.frame(theta2 = y[,2], x)

param.Test <- data.frame(ytest)

colnames(param.Test) <- c("theta1", "theta2")

stats.Test <- data.frame(xtest)

########################################################
############## ABC-RF methodology ######################
########################################################

ncores <- 7 # number of CPU cores to use for parallelization

modelAbcrf.theta1 <- regAbcrf(theta1~., data=data.theta1, ntree=500, paral=TRUE, verbose=FALSE, ncores=ncores)
pred.theta1 <- predict(modelAbcrf.theta1, obs=stats.Test, training=data.theta1, quantiles=c(0.025,0.975), paral=TRUE, ncores=ncores) # predictions
modelAbcrf.theta2 <- regAbcrf(theta2~., data=data.theta2, ntree=500, paral=TRUE, verbose=FALSE, ncores=ncores) #model

pred.theta2 <- predict(modelAbcrf.theta2, obs=stats.Test, training=data.theta2, quantiles=c(0.025,0.975), paral=TRUE, ncores=ncores) # predictions

mean(abs((theta1.test.exact-pred.theta1$expectation)/theta1.test.exact))
mean(abs((theta2.test.exact-pred.theta2$expectation)/theta2.test.exact))

mean(abs((var1.test.exact-pred.theta1$variance)/var1.test.exact))
mean(abs((var2.test.exact-pred.theta2$variance)/var2.test.exact))

mean(abs((quant.theta1.test.freq[,1]-pred.theta1$quantiles[,1])/quant.theta1.test.freq[,1]))
mean(abs((quant.theta2.test.freq[,1]-pred.theta2$quantiles[,1])/quant.theta2.test.freq[,1]))

mean(abs((quant.theta1.test.freq[,2]-pred.theta1$quantiles[,2])/quant.theta1.test.freq[,2]))
mean(abs((quant.theta2.test.freq[,2]-pred.theta2$quantiles[,2])/quant.theta2.test.freq[,2]))

########################################################
############## ABC rejection sampler ###################
########################################################

para.simu <- cbind(theta1.train, theta2.train)
colnames(para.simu) <- c("theta1", "theta2")

cl <- makeCluster(ncores)
registerDoParallel(cl)

matAbcRejet <- foreach(i=1:p, .combine='cbind', .packages="abc") %dopar%{
  res.rejection <- abc(xtest[i,], para.simu, x, tol=0.01, method="rejection")
  c(mean(res.rejection$unadj.values[,1]), mean(res.rejection$unadj.values[,2]),
    var(res.rejection$unadj.values[,1]), var(res.rejection$unadj.values[,2]),
    quantile(res.rejection$unadj.values[,1], 0.025), quantile(res.rejection$unadj.values[,1], 0.975),
    quantile(res.rejection$unadj.values[,2], 0.025), quantile(res.rejection$unadj.values[,2], 0.975) )
}

stopCluster(cl)

theta1.abc.reject <- matAbcRejet[1,]
theta2.abc.reject <- matAbcRejet[2,]
var1.abc.reject <- matAbcRejet[3,]
var2.abc.reject <- matAbcRejet[4,]
quant.theta1.abc.reject <- cbind(matAbcRejet[5,], matAbcRejet[6,] )
quant.theta2.abc.reject <- cbind(matAbcRejet[7,], matAbcRejet[8,] )

mean(abs((theta1.abc.reject-theta1.test.exact)/theta1.test.exact))
mean(abs((theta2.abc.reject-theta2.test.exact)/theta2.test.exact))

mean(abs((var1.abc.reject-var1.test.exact)/var1.test.exact))
mean(abs((var2.abc.reject-var2.test.exact)/var2.test.exact))

mean(abs((quant.theta1.abc.reject[,1]-quant.theta1.test.freq[,1])/quant.theta1.test.freq[,1]))
mean(abs((quant.theta2.abc.reject[,1]-quant.theta2.test.freq[,1])/quant.theta2.test.freq[,1]))

mean(abs((quant.theta1.abc.reject[,2]-quant.theta1.test.freq[,2])/quant.theta1.test.freq[,2]))
mean(abs((quant.theta2.abc.reject[,2]-quant.theta2.test.freq[,2])/quant.theta2.test.freq[,2]))

########################################################
############## Local linear ABC adjustment #############
########################################################

cl <- makeCluster(ncores)
registerDoParallel(cl)

matAbcLoclin <- foreach(i=1:p, .combine='cbind', .packages="abc") %dopar%{
  res.loclin <- abc(xtest[i,], para.simu, x, tol=0.1, method="loclinear")
  c(mean(res.loclin$adj.values[,1]), mean(res.loclin$adj.values[,2]),
    var(res.loclin$adj.values[,1]), var(res.loclin$adj.values[,2]),
    quantile(res.loclin$adj.values[,1], 0.025), quantile(res.loclin$adj.values[,1], 0.975),
    quantile(res.loclin$adj.values[,2], 0.025), quantile(res.loclin$adj.values[,2], 0.975) )
}

stopCluster(cl)

theta1.abc.loclin <- matAbcLoclin[1,]
theta2.abc.loclin <- matAbcLoclin[2,]
var1.abc.loclin <- matAbcLoclin[3,]
var2.abc.loclin <- matAbcLoclin[4,]
quant.theta1.abc.loclin <- cbind(matAbcLoclin[5,], matAbcLoclin[6,] )
quant.theta2.abc.loclin <- cbind(matAbcLoclin[7,], matAbcLoclin[8,] )

mean(abs((theta1.abc.loclin-theta1.test.exact)/theta1.test.exact))
mean(abs((theta2.abc.loclin-theta2.test.exact)/theta2.test.exact))

mean(abs((var1.abc.loclin-var1.test.exact)/var1.test.exact))
mean(abs((var2.abc.loclin-var2.test.exact)/var2.test.exact))

mean(abs((quant.theta1.abc.loclin[,1]-quant.theta1.test.freq[,1])/quant.theta1.test.freq[,1]))
mean(abs((quant.theta2.abc.loclin[,1]-quant.theta2.test.freq[,1])/quant.theta2.test.freq[,1]))

mean(abs((quant.theta1.abc.loclin[,2]-quant.theta1.test.freq[,2])/quant.theta1.test.freq[,2]))
mean(abs((quant.theta2.abc.loclin[,2]-quant.theta2.test.freq[,2])/quant.theta2.test.freq[,2]))

########################################################
############## Ridge ABC adjustment ####################  
########################################################

cl <- makeCluster(ncores)
registerDoParallel(cl)

matAbcRidge<- foreach(i=1:p, .combine='cbind', .packages="abc") %dopar%{
  res.ridge <- abc(xtest[i,], para.simu, x, tol=0.1, method="ridge")
  c(mean(res.ridge$adj.values[,1]), mean(res.ridge$adj.values[,2]),
    var(res.ridge$adj.values[,1]), var(res.ridge$adj.values[,2]),
    quantile(res.ridge$adj.values[,1], 0.025), quantile(res.ridge$adj.values[,1], 0.975),
    quantile(res.ridge$adj.values[,2], 0.025), quantile(res.ridge$adj.values[,2], 0.975) )
}

stopCluster(cl)

theta1.abc.ridge <- matAbcRidge[1,]
theta2.abc.ridge <- matAbcRidge[2,]
var1.abc.ridge <- matAbcRidge[3,]
var2.abc.ridge <- matAbcRidge[4,]
quant.theta1.abc.ridge <- cbind(matAbcRidge[5,], matAbcRidge[6,] )
quant.theta2.abc.ridge <- cbind(matAbcRidge[7,], matAbcRidge[8,] )

mean(abs((theta1.abc.ridge-theta1.test.exact)/theta1.test.exact))
mean(abs((theta2.abc.ridge-theta2.test.exact)/theta2.test.exact))

mean(abs((var1.abc.ridge-var1.test.exact)/var1.test.exact))
mean(abs((var2.abc.ridge-var2.test.exact)/var2.test.exact))

mean(abs((quant.theta1.abc.ridge[,1]-quant.theta1.test.freq[,1])/quant.theta1.test.freq[,1]))
mean(abs((quant.theta2.abc.ridge[,1]-quant.theta2.test.freq[,1])/quant.theta2.test.freq[,1]))

mean(abs((quant.theta1.abc.ridge[,2]-quant.theta1.test.freq[,2])/quant.theta1.test.freq[,2]))
mean(abs((quant.theta2.abc.ridge[,2]-quant.theta2.test.freq[,2])/quant.theta2.test.freq[,2]))

########################################################
############## Neural network ABC adjustment ###########  
########################################################

cl <- makeCluster(ncores)
registerDoParallel(cl)

matAbcNeural <- foreach(i=1:p, .combine='cbind', .packages="abc") %dopar%{
  res.neural <- abc(xtest[i,], para.simu, x, tol=0.1, method="neuralnet")
  c(mean(res.neural$adj.values[,1]), mean(res.neural$adj.values[,2]),
    var(res.neural$adj.values[,1]), var(res.neural$adj.values[,2]),
    quantile(res.neural$adj.values[,1], 0.025), quantile(res.neural$adj.values[,1], 0.975),
    quantile(res.neural$adj.values[,2], 0.025), quantile(res.neural$adj.values[,2], 0.975) )
}

stopCluster(cl)

theta1.abc.neural <- matAbcNeural[1,]
theta2.abc.neural <- matAbcNeural[2,]
var1.abc.neural <- matAbcNeural[3,]
var2.abc.neural <- matAbcNeural[4,]
quant.theta1.abc.neural <- cbind(matAbcNeural[5,], matAbcNeural[6,] )
quant.theta2.abc.neural <- cbind(matAbcNeural[7,], matAbcNeural[8,] )

mean(abs((theta1.abc.neural-theta1.test.exact)/theta1.test.exact))
mean(abs((theta2.abc.neural-theta2.test.exact)/theta2.test.exact))

mean(abs((var1.abc.neural-var1.test.exact)/var1.test.exact))
mean(abs((var2.abc.neural-var2.test.exact)/var2.test.exact))

mean(abs((quant.theta1.abc.neural[,1]-quant.theta1.test.freq[,1])/quant.theta1.test.freq[,1]))
mean(abs((quant.theta2.abc.neural[,1]-quant.theta2.test.freq[,1])/quant.theta2.test.freq[,1]))

mean(abs((quant.theta1.abc.neural[,2]-quant.theta1.test.freq[,2])/quant.theta1.test.freq[,2]))
mean(abs((quant.theta2.abc.neural[,2]-quant.theta2.test.freq[,2])/quant.theta2.test.freq[,2]))

#####################################################
############ Sequential ABC methods #################
#####################################################

# The model

myModel <- function(x) {
  y <- rnorm(10, x[1], sqrt(x[2]))
  meany <- mean(y)
  vary <- var(y)
  mady <- mad(y)
  return(
    c(
      meany,
      vary,
      mady,
      runif(nNoise, 0, 1),
      meany + vary,
      meany + mady,
      vary + mady,
      meany + vary + mady,
      meany * vary,
      meany * mady,
      vary * mady,
      meany * vary * mady
    )
  )
}

# The prior

rNormalInverseGamma <- function() {
  theta2 <- 1 / rgamma(1, shape = 4, rate = 3)
  theta1 <- rnorm(1, 0, sqrt(theta2))
  return(c(theta1, theta2))
}

dNormalInverseGamma <- function(x) {
  if (x[2] <= 0) {
    return(0)
  } else{
    return(1 / sqrt(x[2] * 2 * pi) * (3 ^ 4 / gamma(4)) * (1 / x[2]) ^ 5 * exp(-(6 + (x[1]) ^ 2) / (2 * x[2])))
  }
}

# Function to compute some quantities of interest

predictionQoI <- function(res, quantiles=c(0.025,0.975))
{
  posteriorMean <- rep(NA, ncol(res$param))
  posteriorVar <- rep(NA, ncol(res$param))
  posteriorQuantiles <- matrix(NA, ncol=ncol(res$param), nrow=length(quantiles) )
  colnames(posteriorQuantiles) <- quantiles
  
  for(i in 1:ncol(res$param)){
    posteriorMean[i] <- weighted.mean(x = res$param[,i], w = res$weights)
    posteriorVar[i] <- weighted.var(x = res$param[,i], w = res$weights)
    posteriorQuantiles[i,] <- weighted.quantile(x = res$param[,i], w = res$weights, probs=quantiles)
  }
  return(list(posteriorMean = posteriorMean, posteriorVar = posteriorVar,
              posteriorQuantiles=posteriorQuantiles, nbrReqSimu = res$nbrReqSimu))
}


#####################################################
##### Del Moral et al. (2012) ABC-SMC ###############
#####################################################

# Compute the ESS

ESS <- function(weights){
  return(  ( sum(weights) )^2 / (sum(weights^2)) )
}

# Compute the updated weights

computeWeights <- function(distance, epsilon_new, weights_old){
  
  weights_new <- rep(NA, length(distance))
  for(i in 1:length(distance)){
    weights_new[i] <- weights_old[i] * (distance[i] <= epsilon_new)
  }
  return(weights_new)
  
}

# Perform the dichotomy to claculate epsilon_n 

dichotomieESS <- function(error_max, a_start, b_start, alpha, ESS_old, dist_old, weights_old, dataFrame){
  
  b <- b_start
  a <- a_start
  while( (b-a)>error_max ){
    
    m <- mean(c(a,b))
    poids_a <- computeWeights(distance = dist_old, epsilon_new = a, weights_old = weights_old)
    toto_a <- aggregate(poids_a, dataFrame, sum)
    ESS_a <- ESS(toto_a$x)
    f_a <- ESS_a - alpha*ESS_old
    
    poids_m <- computeWeights(distance = dist_old, epsilon_new = m, weights_old = weights_old)
    toto_m <- aggregate(poids_m, dataFrame, sum)
    ESS_m <- ESS(toto_m$x)
    f_m <- ESS_m - alpha*ESS_old
    
    if(f_a*f_m<=0){
      b <- m
    } else {
      a <- m
    }
    
  }
  return(b)
  
}

# Function to perform the abc.SMC algorithm

## Input :
# myModel: the function to compute summary statistics
# myPrior: the prior used
# probaPrior: dNormalInverseGamma, providing the density value given simulated parameter values
# nbrParam: number of parameters
# nbrSimu: number of simulated parameter values at each step of the algorithm (number of particles)
# nbrSumsta: number of summary statistics used
# obsSummarized: summary statistics values of the test instance
# epsilon: the final epsilon, taking as stopping rule
# alpha: parameter of the algorithm, set as 0.9
# N_T = nbrSimu/2, as in Del Moral et al. (2012)
# sigmaInit: the initial normalization constants to compute distances (Euclidean)
# We use M=1

abc.SMC <- function(myModel, myPrior, probaPrior, nbrParam, nbrSimu, nbrSumsta, obsSummarized, epsilon, alpha, N_T, sigmaInit){
  
  # Step 0: sample instances according to the prior and assign weights
  matrixParam <- matrix(NA, nrow=nbrSimu, ncol=nbrParam)
  matrixSumsta <- matrix(NA, nrow=nbrSimu, ncol=nbrSumsta)
  dist_old <- rep(NA, nbrSimu)
  
  for(i in 1:nbrSimu){
    matrixParam[i,] <- myPrior()
    matrixSumsta[i,] <- myModel(matrixParam[i,])
    dist_old[i] <- sqrt( mean( (  (matrixSumsta[i,]-obsSummarized)/sigmaInit )^2 ) )
  }
  
  # nbrReqSimu compute the total number of simulations used during the algorithm
  nbrReqSimu <- nbrSimu
  
  # tau2 is used in step 3, when simulations are made according to a Kernel, here Gaussian, not weighted yet
  tau2 <- 2*cov(matrixParam)
  
  weights_old <- rep(1, nbrSimu)
  ESS_old <- ESS(weights_old)
  epsilon_old <- Inf
  epsilon_new <- epsilon_old
  
  b_start <- max(dist_old)
  a_start <- min(dist_old)
  
  while(epsilon_new != epsilon){
    
    # Step 1: Determine epsilon_n (epsilon_new) by dichotomy and update weights
    error_max <- 0.01
    dataFrame = data.frame(param = matrixParam)
    epsilon_potential <- dichotomieESS(error_max = error_max, a_start = a_start, b_start = b_start, alpha = alpha,
                                       ESS_old = ESS_old, dist_old = dist_old, weights_old = weights_old, dataFrame = dataFrame)
    weights_new <- computeWeights(dist_old, epsilon_potential, weights_old)
    
    # Compute the new ESS value
    tmp_new <- aggregate(weights_new, dataFrame, sum)
    ESS_new <- ESS(tmp_new$x)
    
    # Update epsilon
    if(epsilon_potential < epsilon){
      epsilon_new <- epsilon
    } else{
      epsilon_new <- epsilon_potential
    }
    
    # Step 2: if ESS_new < N_T resample N particles
    if( ESS_new < N_T ){
      nbrReqSimu <- nbrReqSimu + nbrSimu
      sampled <- sample(x = 1:nbrSimu, size = nbrSimu, replace = TRUE, prob = weights_new)
      
      matrixParam <- matrixParam[sampled,]
      matrixSumsta <- matrixSumsta[sampled,]
      weights_new <- rep(1, nbrSimu)
      
      # tau2 is updated and equal to twice the empirical weighted variance-covariance matrix,
      # used later for the simulation process according to K_n - Gaussian Kernel here
      tau2 <- 2*cov.wt(x = matrixParam, wt = weights_new/sum(weights_new))$cov
    }
    
    for(i in 1:nbrSimu){
      
      # Step 3: MCMC step, simulated according to a Gaussian Kernel(theta^i_{n-1},sqrt(tau2))
      if(weights_new[i]>0){
        thetaAlt <- matrixParam[i,] + rmvnorm(1, c(0,0), tau2)
        nbrReqSimu <- nbrReqSimu + 1
        priorThetaAlt <- probaPrior(thetaAlt)
        priorThetaOld <- probaPrior(matrixParam[i,])
        
        if(priorThetaAlt==0){
          probaAcceptation <- 0
        } else {
          sumstaAlt <- myModel(thetaAlt)
          distAlt <- sqrt( mean( ( (sumstaAlt-obsSummarized)/sigmaInit)^2 ) )
          probaAcceptation <- min(1, (priorThetaAlt/priorThetaOld) * (distAlt<epsilon_new) ) 
        }
        
        if(runif(1) < probaAcceptation){
          matrixParam[i,] <- thetaAlt
          matrixSumsta[i,] <- sumstaAlt
        }
      }
    }
    tau2 <- 2*cov.wt(x = matrixParam, wt = weights_new/sum(weights_new))$cov
    epsilon_old <- epsilon_new
    
    dataFrame = data.frame(param = matrixParam)
    tmp_new <- aggregate(weights_new, dataFrame, sum)
    ESS_new <- ESS(tmp_new$x)
    
    ESS_old <- ESS_new
    dist_old <- sapply(1:nrow(matrixParam), function(x) sqrt( mean( ( (matrixSumsta[x,]-obsSummarized)/sigmaInit )^2 ) ) )
    weights_old <- weights_new
    
    b_start <- epsilon_old
    a_start <- min(dist_old)
  }
  return(list(param=matrixParam, weights = weights_old/sum(weights_old), nbrReqSimu=nbrReqSimu))
}


# First set of calibration parameters

nbrSimu <- 1000
nbrParam <- 2
nbrSumsta <- nNoise+11
myPrior <- rNormalInverseGamma
probaPrior <- dNormalInverseGamma
myModel <- myModel
alpha <- 0.90 
N_T <- nbrSimu/2 
sigmaInit <- apply(x, 2, mad) 
quantOrder <- 0.1

# Parallelisation according to the test dataset

cl <- makeCluster(ncores)
registerDoParallel(cl)

pred.SMC <- foreach(i=1:p, .combine='rbind', .packages=c("spatstat", "MCMCpack", "mvtnorm")) %dopar%{
  
  train.dist <-
    sapply(1:nrow(x), function(X)
      sqrt(mean((( x[X, ] - xtest[i, ]) / sigmaInit) ^ 2)))
  epsilonFinal <- quantile(train.dist, quantOrder)
  res.SMC <- abc.SMC(myModel = myModel, myPrior =  myPrior, probaPrior =  probaPrior, nbrParam = nbrParam, nbrSimu = nbrSimu,
                     nbrSumsta = nbrSumsta, obsSummarized =  xtest[i, ], epsilon = epsilonFinal, alpha = alpha, N_T = N_T, sigmaInit = sigmaInit)
  pred.SMC <- predictionQoI(res.SMC)
  return(c(pred.SMC$posteriorMean, pred.SMC$posteriorVar, pred.SMC$posteriorQuantiles[1,],
           pred.SMC$posteriorQuantiles[2,], pred.SMC$nbrReqSimu))
}

stopCluster(cl)

colnames(pred.SMC) <- c("exp1", "exp2", "var1", "var2", "q1.0.025", "q1.0.975", "q2.0.025", "q2.0.975", "nbrSimu")

mean(abs((pred.SMC[,1]-theta1.test.exact)/theta1.test.exact))
mean(abs((pred.SMC[,2]-theta2.test.exact)/theta2.test.exact))

mean(abs((pred.SMC[,3]-var1.test.exact)/var1.test.exact))
mean(abs((pred.SMC[,4]-var2.test.exact)/var2.test.exact))

mean(abs((pred.SMC[,5]-quant.theta1.test.freq[,1])/quant.theta1.test.freq[,1]))
mean(abs((pred.SMC[,7]-quant.theta2.test.freq[,1])/quant.theta2.test.freq[,1]))

mean(abs((pred.SMC[,6]-quant.theta1.test.freq[,2])/quant.theta1.test.freq[,2]))
mean(abs((pred.SMC[,8]-quant.theta2.test.freq[,2])/quant.theta2.test.freq[,2]))

mean(pred.SMC[,9])

# Second set of calibration parameters

nbrSimu <- 1000 
nbrParam <- 2
nbrSumsta <- nNoise+11
myPrior <- rNormalInverseGamma
probaPrior <- dNormalInverseGamma
myModel <- myModel
alpha <- 0.90
N_T <- nbrSimu/2
sigmaInit <- apply(x, 2, mad)
quantOrder <- 0.01 

# Parallelisation according to the test dataset

cl <- makeCluster(ncores)
registerDoParallel(cl)

pred.SMC <- foreach(i=1:p, .combine='rbind', .packages=c("spatstat", "MCMCpack", "mvtnorm")) %dopar%{
  
  train.dist <-
    sapply(1:nrow(x), function(X)
      sqrt(mean((( x[X, ] - xtest[i, ]) / sigmaInit) ^ 2)))
  epsilonFinal <- quantile(train.dist, quantOrder)
  res.SMC <- abc.SMC(myModel = myModel, myPrior =  myPrior, probaPrior =  probaPrior, nbrParam = nbrParam, nbrSimu = nbrSimu,
                     nbrSumsta = nbrSumsta, obsSummarized =  xtest[i, ], epsilon = epsilonFinal, alpha = alpha, N_T = N_T, sigmaInit = sigmaInit)
  pred.SMC <- predictionQoI(res.SMC)
  return(c(pred.SMC$posteriorMean, pred.SMC$posteriorVar, pred.SMC$posteriorQuantiles[1,],
           pred.SMC$posteriorQuantiles[2,], pred.SMC$nbrReqSimu))
}

stopCluster(cl)

colnames(pred.SMC) <- c("exp1", "exp2", "var1", "var2", "q1.0.025", "q1.0.975", "q2.0.025", "q2.0.975", "nbrSimu")

mean(abs((pred.SMC[,1]-theta1.test.exact)/theta1.test.exact))
mean(abs((pred.SMC[,2]-theta2.test.exact)/theta2.test.exact))

mean(abs((pred.SMC[,3]-var1.test.exact)/var1.test.exact))
mean(abs((pred.SMC[,4]-var2.test.exact)/var2.test.exact))

mean(abs((pred.SMC[,5]-quant.theta1.test.freq[,1])/quant.theta1.test.freq[,1]))
mean(abs((pred.SMC[,7]-quant.theta2.test.freq[,1])/quant.theta2.test.freq[,1]))

mean(abs((pred.SMC[,6]-quant.theta1.test.freq[,2])/quant.theta1.test.freq[,2]))
mean(abs((pred.SMC[,8]-quant.theta2.test.freq[,2])/quant.theta2.test.freq[,2]))

mean(pred.SMC[,9])

#####################################################
##### ABC-PMC from  Beaumont et al. (2009) ##########
#####################################################

# Function to perform the abc.PMC (Beaumont et al. (2009)) algorithm

## Input :
# myModel: the function to compute summary statistics
# myPrior: the prior used
# probaPrior: dNormalInverseGamma, providing the density values given simulated parameter values
# nbrParam: number of parameters
# nbrSimu: number of simulated parameter values at each step of the algorithm
# obsSummarized: summary statistics values of the test instance
# sigmaInit: the initial normalization constant to compute distance
# alphaPercent: the quantile order under which we accept simulated values
# nbrIteration: the number of iterations
# We use M=1 

abc.PMC <- function(myModel, myPrior, probaPrior, nbrParam, nbrSimu,
                    obsSummarized, sigmaInit, alphaPercent, nbrIteration){
  
  nbrReqSimu <- 0
  old_weights <- rep(NA, nbrSimu)
  new_weightsStd <- rep(NA, nbrSimu)
  
  matrixParam <- matrix(NA, ncol = nbrParam, nrow = nbrSimu)
  matrixSumsta <- matrix(NA, ncol = length(obsSummarized), nrow = nbrSimu)
  
  for(i in 1:nbrSimu){
    matrixParam[i,] <- myPrior()
    matrixSumsta[i,] <- myModel(matrixParam[i,])
    nbrReqSimu <- nbrReqSimu+1
  }
  
  Nalpha <- floor(nbrSimu*alphaPercent)
  dist_old <- sapply(1:nbrSimu, function(X) sqrt( mean( ( (matrixSumsta[X,] - obsSummarized)/sigmaInit )^2 ) ) ) 
  orderIndex <- order(dist_old)[1:Nalpha]
  accepted_parameters <- matrixParam[orderIndex,]
  accepted_summary <- matrixSumsta[orderIndex,]
  old_weights <- rep(1/Nalpha, Nalpha)
  tau2 <- 2*cov(accepted_parameters)
  new_parameters <- matrix(NA, ncol=nbrParam, nrow=nbrSimu)
  new_summary <- matrix(NA, ncol=length(obsSummarized), nrow=nbrSimu)
  
  for(k in 2:nbrIteration){
    for(i in 1:nbrSimu){
      
      whichIsPicked <- sample(x=1:Nalpha, size=1, prob=old_weights)
      thetaAlt <- accepted_parameters[whichIsPicked,] + rmvnorm(1, c(0,0), tau2)
      nbrReqSimu <- nbrReqSimu+1
      while(probaPrior(thetaAlt) == 0){
        whichIsPicked <- sample(x=1:Nalpha, size=1, prob=old_weights)
        thetaAlt <- accepted_parameters[whichIsPicked,] + rmvnorm(1, c(0,0), tau2)
      }
      summaryAlt <- myModel(thetaAlt)
      new_parameters[i,] <- thetaAlt
      new_summary[i,] <- summaryAlt
      new_weightsStd[i] <- probaPrior(thetaAlt)/sum(old_weights * sapply(1:length(old_weights), FUN = function(X) dmvnorm(thetaAlt, mean = accepted_parameters[X,], sigma=tau2)))
    }
    
    dist_old <- sapply(1:nbrSimu, function(X) sqrt( mean( ( (new_summary[X,] - obsSummarized)/sigmaInit )^2 ) ) )
    orderIndex <- order(dist_old)[1:Nalpha]
    accepted_parameters <- new_parameters[orderIndex,]
    accepted_summary <- new_summary[orderIndex,]
    old_weights <- new_weightsStd[orderIndex]
    old_weights <- old_weights/sum(old_weights)
    tau2 <- 2*cov.wt(x = accepted_parameters, wt = old_weights)$cov
  }
  return(list(param=accepted_parameters, weights=old_weights, nbrReqSimu=nbrReqSimu))
}

# First set of calibration parameters

nbrSimu <- 1000
nbrIteration <- 10
alphaPercent <- 0.1
sigmaInit <- apply(x, 2, mad)

# Parallelisation according to the test dataset

cl <- makeCluster(ncores)
registerDoParallel(cl)
pred.PMC <- foreach(i=1:p, .combine='rbind', .packages=c("spatstat", "MCMCpack","mvtnorm")) %dopar%{
  res.PMC <- abc.PMC(myModel = myModel,
                     myPrior = rNormalInverseGamma,
                     probaPrior = dNormalInverseGamma,
                     nbrParam = 2,
                     nbrSimu = nbrSimu,
                     obsSummarized = xtest[i, ],
                     sigmaInit = sigmaInit,
                     alphaPercent = alphaPercent, nbrIteration = nbrIteration)
  pred.PMC <- predictionQoI(res.PMC)
  return(c(pred.PMC$posteriorMean, pred.PMC$posteriorVar, pred.PMC$posteriorQuantiles[1,],
           pred.PMC$posteriorQuantiles[2,], pred.PMC$nbrReqSimu))
}

stopCluster(cl)
colnames(pred.PMC) <- c("exp1", "exp2", "var1", "var2", "q1.0.025", "q1.0.975", "q2.0.025", "q2.0.975", "nbrSimu")

mean(abs((pred.PMC[,1]-theta1.test.exact)/theta1.test.exact))
mean(abs((pred.PMC[,2]-theta2.test.exact)/theta2.test.exact))

mean(abs((pred.PMC[,3]-var1.test.exact)/var1.test.exact))
mean(abs((pred.PMC[,4]-var2.test.exact)/var2.test.exact))

mean(abs((pred.PMC[,5]-quant.theta1.test.freq[,1])/quant.theta1.test.freq[,1]))
mean(abs((pred.PMC[,7]-quant.theta2.test.freq[,1])/quant.theta2.test.freq[,1]))

mean(abs((pred.PMC[,6]-quant.theta1.test.freq[,2])/quant.theta1.test.freq[,2]))
mean(abs((pred.PMC[,8]-quant.theta2.test.freq[,2])/quant.theta2.test.freq[,2]))

mean(pred.PMC[,9])

# Second set of calibration parameters

nbrSimu <- 100
nbrIteration <- 100
alphaPercent <- 0.1
sigmaInit <- apply(x, 2, mad)

# Parallelisation according to the test dataset

cl <- makeCluster(ncores)
registerDoParallel(cl)
pred.PMC <- foreach(i=1:p, .combine='rbind', .packages=c("spatstat", "MCMCpack","mvtnorm")) %dopar%{
  res.PMC <- abc.PMC(myModel = myModel,
                     myPrior = rNormalInverseGamma,
                     probaPrior = dNormalInverseGamma,
                     nbrParam = 2,
                     nbrSimu = nbrSimu,
                     obsSummarized = xtest[i, ],
                     sigmaInit = sigmaInit,
                     alphaPercent = alphaPercent, nbrIteration = nbrIteration)
  pred.PMC <- predictionQoI(res.PMC)
  return(c(pred.PMC$posteriorMean, pred.PMC$posteriorVar, pred.PMC$posteriorQuantiles[1,],
           pred.PMC$posteriorQuantiles[2,], pred.PMC$nbrReqSimu))
}

stopCluster(cl)
colnames(pred.PMC) <- c("exp1", "exp2", "var1", "var2", "q1.0.025", "q1.0.975", "q2.0.025", "q2.0.975", "nbrSimu")

mean(abs((pred.PMC[,1]-theta1.test.exact)/theta1.test.exact))
mean(abs((pred.PMC[,2]-theta2.test.exact)/theta2.test.exact))

mean(abs((pred.PMC[,3]-var1.test.exact)/var1.test.exact))
mean(abs((pred.PMC[,4]-var2.test.exact)/var2.test.exact))

mean(abs((pred.PMC[,5]-quant.theta1.test.freq[,1])/quant.theta1.test.freq[,1]))
mean(abs((pred.PMC[,7]-quant.theta2.test.freq[,1])/quant.theta2.test.freq[,1]))

mean(abs((pred.PMC[,6]-quant.theta1.test.freq[,2])/quant.theta1.test.freq[,2]))
mean(abs((pred.PMC[,8]-quant.theta2.test.freq[,2])/quant.theta2.test.freq[,2]))

mean(pred.PMC[,9])

########################################################
####### Adaptive ABC-PMC from Prangle (2017) ###########
########################################################

#### Implementation of the adaptive ABC-PMC algorithm (Prange, 2017)

## Input :
# myModel: the function to compute summary statistics
# myPrior: the prior used
# probaPrior: dNormalInverseGamma, providing the density values given simulated parameter values
# nbrParam: number of parameters
# N: number of accepted simulations per iteration
# alpha: an integer value, set as 0.5 (as in Prangle (2017))
# NToComputeMad: the number of data used to compute the mad during the algorithm among the previous simulations (included those rejected),
# Nstop: the total number of simulation to use before ending the algorithm (the algorithm ends after the current iteration, after this number is reached),
# obsSummarized: summary statistics values of the test instance

abc.PMC.adapt <- function(myModel, myPrior, probaPrior, nbrParam, N, alpha, NToComputeMad, Nstop, obsSummarized){
  
  nbrReqSimu <- 0
  M <- ceiling(N/alpha)
  NCalculMad <- NToComputeMad
  old_weights <- rep(NA, N)
  new_weightsStd <- rep(NA, N)
  
  matrixSumstaCalculMad <- matrix(NA, ncol = length(obsSummarized), nrow = NCalculMad)
  matrixSumstaTempoM <- matrix(NA, ncol = length(obsSummarized), nrow = M)
  matrixParamTempoM <- matrix(NA, ncol = nbrParam, nrow = M)
  accepted_sumsta <- matrix(NA, ncol = length(obsSummarized), nrow = N)
  accepted_parameters <- matrix(NA, ncol = nbrParam, nrow = N)
  previous_accepted_parameters <- matrix(NA, ncol = nbrParam, nrow = N)
  previousMadForDistance <- NULL
  previousDistance <- NULL
  
  noIter <- 1
  
  # Stopping rule
  while(nbrReqSimu < Nstop){
    
    if(noIter == 1){
      for(i in 1:M){
        matrixParamTempoM[i,] <- myPrior()
        matrixSumstaTempoM[i,] <- myModel(matrixParamTempoM[i,])
        nbrReqSimu <- nbrReqSimu+1 
        
        # We store NCalculMad obervations, rejected or not to compute the MAD
        if(i <= NCalculMad){
          matrixSumstaCalculMad[i,] <- matrixSumstaTempoM[i,]
        }
      }
    } else {
      idxCalculMad <- 1 
      for(i in 1:M){
        accepted <- FALSE
        while(!accepted){
          whichIsPicked <- sample(x=1:N, size=1, prob=old_weights)
          thetaAlt <- previous_accepted_parameters[whichIsPicked,] + rmvnorm(1, c(0,0), tau2) 
          while(dNormalInverseGamma(thetaAlt) == 0){
            whichIsPicked <- sample(x=1:N, size=1, prob=old_weights)
            thetaAlt <- previous_accepted_parameters[whichIsPicked,] + rmvnorm(1, c(0,0), tau2)
          }
          summaryAlt <- myModel(thetaAlt)
          nbrReqSimu <- nbrReqSimu+1 
          if(idxCalculMad <= NCalculMad){
            matrixSumstaCalculMad[idxCalculMad,] <- summaryAlt
            idxCalculMad <- idxCalculMad + 1
          }
          matchDist <- sapply(1:(noIter-1), function(X) sqrt( mean( ( (summaryAlt-obsSummarized)/previousMadForDistance[X,] )^2 ) ) <= previousDistance[X] )
          if(all(matchDist)){
            accepted <- TRUE
            matrixParamTempoM[i,] <- thetaAlt
            matrixSumstaTempoM[i,] <- summaryAlt
          }
        }
      }
    }
    
    madIterationt <- apply(matrixSumstaCalculMad, 2, mad)
    previousMadForDistance <- rbind(previousMadForDistance, madIterationt)
    distance_M <- sapply(1:M, function(X) sqrt( mean( ( (matrixSumstaTempoM[X,]-obsSummarized)/madIterationt )^2 ) ) )
    idxOrder <- order(distance_M)
    h_t <- distance_M[idxOrder[N]]
    previousDistance[noIter] <- h_t
    
    # We keep the Nth parameters as accepted among the M
    idxOrderToKeepForN <- idxOrder[1:N]
    accepted_parameters <- matrixParamTempoM[idxOrderToKeepForN,]
    if(noIter==1){
      new_weights <- rep(1/N,N)
    } else {
      new_weights <- rep(NA, N)
      for(j in 1:N){
        new_weights[j] <- probaPrior(accepted_parameters[j,])/sum(old_weights * sapply(1:N, FUN = function(X) dmvnorm(accepted_parameters[j,], mean = previous_accepted_parameters[X,], sigma=tau2)))
      }
      new_weights <- new_weights/sum(new_weights)
    }
    noIter <- noIter+1
    old_weights <- new_weights
    previous_accepted_parameters <- accepted_parameters
    tau2 <- 2*cov.wt(x = previous_accepted_parameters, wt = old_weights)$cov
  }
  return(list(param=previous_accepted_parameters, weights=old_weights, nbrReqSimu=nbrReqSimu, noIter=noIter-1))
}

# First set of calibration parameters

N <- 1000
Nstop <- 10000
NToComputeMad <- 1000
alpha <- 0.5

# Parallelisation according to the test dataset

cl <- makeCluster(ncores)
registerDoParallel(cl)

pred.PMC <- foreach(i=1:p, .combine='rbind', .packages=c("spatstat", "MCMCpack","mvtnorm")) %dopar%{
  res.PMC <- abc.PMC.adapt(myModel = myModel, myPrior = rNormalInverseGamma,
                           probaPrior = dNormalInverseGamma, nbrParam = 2,
                           N = N, alpha = alpha, Nstop = Nstop, obsSummarized = xtest[i,], NToComputeMad = NToComputeMad)
  pred.PMC <- predictionQoI(res.PMC)
  return(c(pred.PMC$posteriorMean, pred.PMC$posteriorVar, pred.PMC$posteriorQuantiles[1,],
           pred.PMC$posteriorQuantiles[2,], pred.PMC$nbrReqSimu))
}
stopCluster(cl)

colnames(pred.PMC) <- c("exp1", "exp2", "var1", "var2", "q1.0.025", "q1.0.975", "q2.0.025", "q2.0.975", "nbrSimu")

mean(abs((pred.PMC[,1]-theta1.test.exact)/theta1.test.exact))
mean(abs((pred.PMC[,2]-theta2.test.exact)/theta2.test.exact))

mean(abs((pred.PMC[,3]-var1.test.exact)/var1.test.exact))
mean(abs((pred.PMC[,4]-var2.test.exact)/var2.test.exact))

mean(abs((pred.PMC[,5]-quant.theta1.test.freq[,1])/quant.theta1.test.freq[,1]))
mean(abs((pred.PMC[,7]-quant.theta2.test.freq[,1])/quant.theta2.test.freq[,1]))

mean(abs((pred.PMC[,6]-quant.theta1.test.freq[,2])/quant.theta1.test.freq[,2]))
mean(abs((pred.PMC[,8]-quant.theta2.test.freq[,2])/quant.theta2.test.freq[,2]))

mean(pred.PMC[,9])

# Second set of calibration parameters

N <- 1000
Nstop <- 100000
NToComputeMad <- 1000
alpha <- 0.5

# Parallelisation according to the test dataset

cl <- makeCluster(ncores)
registerDoParallel(cl)

pred.PMC <- foreach(i=1:p, .combine='rbind', .packages=c("spatstat", "MCMCpack","mvtnorm")) %dopar%{
  res.PMC <- abc.PMC.adapt(myModel = myModel, myPrior = rNormalInverseGamma,
                           probaPrior = dNormalInverseGamma, nbrParam = 2,
                           N = N, alpha = alpha, Nstop = Nstop, obsSummarized = xtest[i,], NToComputeMad = NToComputeMad)
  pred.PMC <- predictionQoI(res.PMC)
  return(c(pred.PMC$posteriorMean, pred.PMC$posteriorVar, pred.PMC$posteriorQuantiles[1,],
           pred.PMC$posteriorQuantiles[2,], pred.PMC$nbrReqSimu))
}
stopCluster(cl)

colnames(pred.PMC) <- c("exp1", "exp2", "var1", "var2", "q1.0.025", "q1.0.975", "q2.0.025", "q2.0.975", "nbrSimu")

mean(abs((pred.PMC[,1]-theta1.test.exact)/theta1.test.exact))
mean(abs((pred.PMC[,2]-theta2.test.exact)/theta2.test.exact))

mean(abs((pred.PMC[,3]-var1.test.exact)/var1.test.exact))
mean(abs((pred.PMC[,4]-var2.test.exact)/var2.test.exact))

mean(abs((pred.PMC[,5]-quant.theta1.test.freq[,1])/quant.theta1.test.freq[,1]))
mean(abs((pred.PMC[,7]-quant.theta2.test.freq[,1])/quant.theta2.test.freq[,1]))

mean(abs((pred.PMC[,6]-quant.theta1.test.freq[,2])/quant.theta1.test.freq[,2]))
mean(abs((pred.PMC[,8]-quant.theta2.test.freq[,2])/quant.theta2.test.freq[,2]))

mean(pred.PMC[,9])