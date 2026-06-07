#SMC for Toy Model
library(MASS)
library(abcsmcrf)
library(invgamma)
library(parallel)
set.seed(1)
n_cores <- parallel::detectCores() - 3
#in alg smcdrf: 
#T = n_gen (numero di generazioni),
#N_t= n_simulazioni per generazione
n_gen <- 5
n_sim <- 100 #in drf era 10000, quindi valuta quanto in funzione di n_gen 
a0 <- 4 #shape
b0 <- 3 #rate
n_oss <- 10
sst<- function(y){
  Ybar <- mean(y)
  SD <- sd(y)
  MAD <- mad(y)
  
  c(
    Ybar=Ybar, 
    SD=SD,
    MAD=MAD,
    SQ=sum((y-Ybar)^2),
    #Ybar e SQ sono statistiche sufficienti per th1 e th2, ne mettiamo di piu per dimostrare che la drf è capace di ignorare quelle inutili
    Median=median(y),
    q25=quantile(y, 0.25),
    q90=quantile(y, 0.9),
    q75=quantile(y, 0.75),
    skew=mean((y-Ybar)^3/SD^3))
}

#Da predirre:
th2_obs <- rinvgamma(1, shape=a0, rate=b0)
th1_obs <-rnorm(1, mean = 0, sd=sqrt(th2_obs))
y_obs <- rnorm(n_oss, mean = th1_obs, sd=sqrt(th2_obs))
s_obs <- c(sst(y_obs), runif(50))
#abcsmcrf vuole i dati obs come data.frame quindi
s_obs_smc <- as.data.frame(t(s_obs))
colnames(s_obs_smc) <- c("Mean", "SD", "MAD", "SQ", 
                         "Median", "quantile 25", 
                         "quantile 90", "quantile 75",
                         "skewness", paste0("noise #", 1:50))

cat(sprintf("Parameters to predict: th1= %.3f, th2= %.3f", th1_obs, th2_obs))

#funzione che prende ogni volta i theta proposi da kernel generazione t,
#simula n_oss osservazioni e calcola le summary stats
modello <-  function(theta_reference){
  n_sim <- nrow(theta_reference)
  #questa sotto è la parte s^i della ref table 
  #ma cambia ad ogni generazione avvicinandosi a quella reale
  X_ref <- matrix(NA, nrow = n_sim, ncol=ncol(s_obs_smc))
  
  for (i in 1:n_sim) {
    y_sim <- rnrom(n_oss, mean=theta_reference$th1[i], sd=sqrt(theta_reference$th2[i]))
    X_ref[i,] <- c(sst(y_sim), runif(50))}
    Reft <- cbind(theta_reference, X_ref)
    colnames(Reft) <- c("theta1", "theta2", colnames(s_obs_smc))
    return(as.data.frame(Reft))
  }

#Prior per la generazione t=1
rprior_smc <- function(num_parametri) {
  th2 <- rinvgamma(num_parametri, shape = a0, rate = b0)
  th1 <- rnorm(num_parametri, mean = 0, sd = sqrt(th2))
  return(data.frame(theta1 = th1, theta2 = th2))
}

#Densita priro per le altre generazioni
#attenziona a "all" e %in% e i due if dopo perchè si potrebbero togliere 
#ma li teniamo per le marginali
dprior_smc <- function(theta, dens="all") #all significa joint density in automatico, senno posso fare anche solo th1/2
  probs <- rep(1, nrow(theta)) #metto 1 tanto sto facendo moltiplicazioni
  vec_val
  















  