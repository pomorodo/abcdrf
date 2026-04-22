#sapply, outer, funzioni kde e drf, bw=SJ, Vectorize
#t-student
#library(abcrf) 
library(MASS) #per kde
library(invgamma)
library(drf) #per drf e predict (drf objects)
library(parallel)#per tutte le robe multicore
set.seed(1)
n_cores <- parallel::detectCores() - 2 #per il mio mac è la funzione più safe
# Parametri del modello per il TRAINING
a0<-4 #shape della IG
b0<-3 #rate della IG
n_oss<-10  # del campione

# Summary Stat., il paper usa 11 features ridondanti: mean, sd, MAD e sum/prod tra essi
# Qui ne propongo altri, poichè vorrei mostrarne l'utilità, mentre Raynal voleva
# mostrarne il potere decisionale

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
} #11 features
cat("Summ Stats...\n")
#   Ref set tale per cui
#   Y_i | th  ~ N(th1, th2) (ll)
#   th1  | th2 ~ N(0, th2) (prior su th1 cond a th2)
#   th2 ~ IG(a0, b0)  (prior marg su th2)
n_sim<-10000
th2<- rinvgamma(n_sim, shape = a0, rate = b0)
th1<- rnorm(n_sim, mean = 0, sd=sqrt(th2))
cat("X_ref...\n") #fai riferimento ad algortihm1 (Raynal)
#voglio simulare Y come matrice n_simXn_oss 
# cosicchè su ogni riga i ho le osservazioni data una coppia theta i fissata per quella riga
Y<-matrix(nrow = n_sim, ncol= n_oss)
#mclapply è la verisone multicore parallel di sapply (non ha piu bisogno della trasposta)
Y<-mclapply(1:n_sim, function(i){
  rnorm(n_oss, mean = th1[i], sqrt(th2[i]))
})
Y <- do.call(rbind,Y) #da questa Y, voglio che ogni riga diventi summ.stat.tra le n_oss osservazioni, 
#cioè voglio n_sim summary statistics:
#Sref è la sst di Y riga per riga ma con solo le feature che ho ocstruito io, Xref sarà quella con e feature noisy
Sref<-mclapply(1:n_sim,
               function(r){
                 sst(Y[r,])
               }, mc.cores = n_cores)#mclapply la trasforma in lista, quindi
Sref<-do.call(rbind,Sref) #perchè hai usato do.call? perchè noi vogliamo un vettore

Rumref<-matrix(runif(50*n_sim), nrow = n_sim, ncol = 50)
colnames(Rumref)<-paste0("feature #", 1:50)

Xref<-cbind(Sref,Rumref)

#Dati Osservati "true"
set.seed(1)
th2_vero<-rinvgamma(1,shape = a0, rate = b0)
th1_vero<-rnorm(1,mean = 0, sd = sqrt(th2_vero))
y_temp<-rnorm(n_oss, mean = th1_vero, sd = sqrt(th2_vero))

y_vero<-c(sst(y_temp), runif(50))
y_vero<-as.data.frame(t(y_vero))
colnames(y_vero) <- colnames(Xref)
cat(sprintf("True parameters: th1 = %.3f   th2 = %.3f\n",th1_vero, th2_vero))

#Posterior esatta (data da Raynal)
ybar_post<-mean(y_temp) 
sq_post<-sum((y_temp - ybar_post)^2)
B<-0.5*(sq_post + 6 + n_oss*ybar_post^2 / (n_oss+1))
# (1) th2 | Y ~ IG(a_post, b_post)
# (2) th1 | Y ~ t_n+8 (mu, tau2)
a_post<- n_oss / 2 + 4
b_post <- B
mu <- n_oss * ybar_post / (n_oss + 1)
tau2 <- 2*B/((n_oss +1)*(n_oss + 8))
#true joint p(th1,th2|y)=p(th1|th2,y)p(th2|y)=sum log
joint_post <- function(theta1, theta2){
  exp(
    dinvgamma(theta2, shape = a_post, rate = b_post, log = TRUE) +  # log p(th2|y)
      dnorm(theta1, mean = mu, sd = sqrt(2 * theta2 / (n_oss + 1)), log = TRUE)  # log p(th1|th2,y)
  )
}
gdl<- n_oss+8 #gradi di libertà
#dens marginali vere
#le uso dopo aver definito gli assi dei tempi

#ABC-DRF
cat("Training DRFs...\n")
#drf(Xref, thjoint, ntrees)
n_trees<-500
drf_tr<-drf(
  X=Xref,
  Y=cbind(theta1=th1, theta2=th2),
  num.trees = n_trees,
  num.threads = n_cores)
#predict ha bisogno di una matrice 1xnfeatures quindi
y_vero_matx<-matrix(as.numeric(y_vero), nrow = 1)
colnames(y_vero_matx) <- colnames(Xref)
drf_pred<-predict(drf_tr, newdata=y_vero_matx)
pesi<-drf_pred$weights[1,]
#normalizzazione e pulizia impurità
#pesi<-pmax(pesi,0)
#pesi<-pesi/sum(pesi) 

#Campionamento joint posterior, approx p(th1,th2|y_vero)
#Sfruttiamo i pesi calcolati dalle drf
sampling<-sample(n_sim, size = 10000, replace = TRUE, prob = pesi )
th1_drf <- th1[sampling]
th2_drf <- th2[sampling]
#densità marginali pesate estratte dalle DRF, invece di calcolarle con abc-rf
th1_densmarg<-density(th1, weights = pesi, n=512, bw="SJ")
th2_densmarg<-density(th2, weights = pesi, n=512, bw="SJ") 
#bandwidth Sheather-Jones (per non avere eccesso smoothing, importante per ig)

#Assi per joint
asse1_joint <- seq(-1,3,length.out=300) #th1 sta tra questi due valori
asse2_joint <- seq(0.01,4,length.out=300)
mappa_joint <- outer(asse1_joint,asse2_joint,Vectorize(joint_post)) #con densità true
#Assi per le marginals
asse1_marg <- seq(-1,3,length.out=600) #th1 sta tra questi due valori
asse2_marg <- seq(0.01,4
                  ,length.out=600)
#True marg
dens1_vera<- dt((asse1_marg-mu)/sqrt(tau2),df=gdl)/sqrt(tau2) #t_n+8(mu, √tau2)
dens2_vera<- dinvgamma(asse2_marg, shape = a_post, rate=b_post)#ig(apost,bpost)

#Figura————————————————————————————————
kde_joint<-kde2d(
  x=th1_drf,
  y=th2_drf,
  n=200,
  lims=c(-1,3,0.01,4)
)
#mappa_joint[mappa_joint > max(mappa_joint) * 0.05] # cosi posso tagliare le code quasi piatte e tenere solo valori rilevanti (controlla se è giusto)
livelli <- seq(min(mappa_joint), max(mappa_joint), length.out = 9)

#Heatmap
image(kde_joint,
      col=hcl.colors(8, "plasma"),
      xlim=c(-1,3),
      ylim=c(0.01,4),
      xlab="Theta 1",
      ylab="Theta 2",
      main="DRF: joint posterior")
contour(asse1_joint, asse2_joint, mappa_joint, add=TRUE, col="white", lty = 6,lwd=1, levels = livelli)
points(th1_vero,th2_vero,lwd=3, pch=4, cex= 2,col="green")
legend("topright",
       legend = c("DRF", "True post", "True (th1, th2)"),
       text.col = "white",
       col    = c("red", "white", "green"),
       lty    = c(1, 1, 1),
       lwd    = c(3, 3, 3),
       bty    = "n")

# Contour corretto?√
#Marginale di th1 ———————————————————
ylim2 <- c(0, max(dens1_vera, th1_densmarg$y) * 1.25)

plot(asse1_marg, dens1_vera,
     type = "l", col = "red", lwd = 2.5,
     xlim = c(-1, 3), ylim = ylim2,
     xlab = "Theta1",
     ylab = "Dens",
     main = "Marginal Theta1")

polygon(c(th1_densmarg$x, rev(th1_densmarg$x)),
        c(th1_densmarg$y, rep(0, length(th1_densmarg$y))),
        col = adjustcolor("steelblue", alpha.f = 0.30),
        border = NA)

lines(th1_densmarg$x, th1_densmarg$y, col = "steelblue", lwd = 2.5)

legend("topright",
       legend = c("True posterior", "ABC-DRF"),
       col    = c("red", "steelblue"),
       lwd    = 2.5, bty = "n")


#Marginale di th2 ———————————————
ylim3 <- c(0, max(dens2_vera, th2_densmarg$y) * 1.25)

plot(asse2_marg, dens2_vera,
     type = "l", col = "red", lwd = 2.5,
     xlim = c(0, 4), ylim = ylim3,
     xlab = "Theta 2",
     ylab = "Density",
     main = "Marginal Theta2")

polygon(c(th2_densmarg$x, rev(th2_densmarg$x)),
        c(th2_densmarg$y, rep(0, length(th2_densmarg$y))),
        col = adjustcolor("steelblue", alpha.f = 0.30),
        border = NA)

lines(th2_densmarg$x, th2_densmarg$y, col = "steelblue", lwd = 2.5)

legend("topright",
       legend = c("True posterior", "ABC-DRF"),
       col    = c("red", "steelblue"),
       lwd    = 2.5, bty = "n", cex = 1.1)







