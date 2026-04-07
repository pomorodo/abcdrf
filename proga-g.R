# ============================================================
#  ABC vs Distributional Random Forests – Versione Ottimizzata
# ============================================================
library(abcrf)
library(drf)
library(invgamma)
library(MASS)

#set.seed(123) # Cambiato seed per una visualizzazione più chiara

# ── 1. PARAMETRI E SIMULAZIONE ──────────────────────────────────────────────
alpha0 <- 4; beta0 <- 3; n_obs <- 50 
N <- 10000

# Funzione statistiche (immutata, ma essenziale)
sumstats <- function(Y) {
  Ybar <- mean(Y); s <- sd(Y); S2 <- sum((Y - Ybar)^2)
  qs <- as.numeric(quantile(Y, c(0.1, 0.25, 0.75, 0.9)))
  c(Ybar=Ybar, S2=S2, varY=var(Y), medY=median(Y), madY=mad(Y), 
    q10=qs[1], q25=qs[2], q75=qs[3], q90=qs[4],
    skew=mean((Y-Ybar)^3)/s^3, kurt=mean((Y-Ybar)^4)/s^4)
}

# Generazione Reference Set
th2 <- rinvgamma(N, shape = alpha0, rate = beta0)
th1 <- rnorm(N, mean = 0, sd = sqrt(th2))
S_ref <- t(sapply(seq_len(N), function(i) sumstats(rnorm(n_obs, th1[i], sqrt(th2[i])))))
X_ref <- cbind(S_ref, matrix(runif(N * 50), nrow = N)) # 11 stats + 50 noise
colnames(X_ref) <- c(colnames(S_ref), paste0("u", 1:50))

# Dati Osservati (Parametri fissi per scopi didattici)
th1_true <- 1.5; th2_true <- 0.8 # Valori scelti per essere ben visibili nel plot
Y_obs <- rnorm(n_obs, mean = th1_true, sd = sqrt(th2_true))
x_obs <- data.frame(t(c(sumstats(Y_obs), runif(50))))
colnames(x_obs) <- colnames(X_ref)

# ── 2. MODELLI (CORREZIONE ABC-RF) ──────────────────────────────────────────
cat("Training modelli...\n")
# Training ABC-RF
rf_t1 <- regAbcrf(theta1 ~ ., data = data.frame(theta1 = th1, X_ref), ntree = 500, paral = FALSE)
rf_t2 <- regAbcrf(theta2 ~ ., data = data.frame(theta2 = th2, X_ref), ntree = 500, paral = FALSE)

# Estrazione pesi per ABC-RF (Metodo robusto per evitare linee mancanti)
# Usiamo i pesi corretti per calcolare le densità marginali
w1_rf <- predict(rf_t1, x_obs, data.frame(theta1 = th1, X_ref), paral = FALSE)$weights[1,]
w2_rf <- predict(rf_t2, x_obs, data.frame(theta2 = th2, X_ref), paral = FALSE)$weights[1,]

# Training ABC-DRF
drf_fit <- drf(X = X_ref, Y = cbind(th1, th2), num.trees = 500)
w_drf <- predict(drf_fit, newdata = as.matrix(x_obs))$weights[1,]

# ── 3. CALCOLO DENSITÀ E LIMITI ─────────────────────────────────────────────
# Range per i grafici (allargati per scopi didattici)
t1_lim <- c(-0.5, 3.5); t2_lim <- c(0.1, 3.0)
t1g <- seq(t1_lim[1], t1_lim[2], length.out = 200)
t2g <- seq(t2_lim[1], t2_lim[2], length.out = 200)

# Densità Marginali
d1_true <- dt((t1g - (n_obs*mean(Y_obs)/(n_obs+1))) / sqrt((S_ref[1]+6+n_obs*mean(Y_obs)^2/(n_obs+1))/((n_obs+1)*(n_obs+8))), df=n_obs+8) # Semplificata per brevità
# Nota: Uso density() con i pesi estratti dalle RF per coerenza
d1_rf_dens  <- density(th1, weights = w1_rf, from=t1_lim[1], to=t1_lim[2], bw="SJ")
d1_drf_dens <- density(th1, weights = w_drf, from=t1_lim[1], to=t1_lim[2], bw="SJ")
d2_rf_dens  <- density(th2, weights = w2_rf, from=t2_lim[1], to=t2_lim[2], bw="SJ")
d2_drf_dens <- density(th2, weights = w_drf, from=t2_lim[1], to=t2_lim[2], bw="SJ")

# ── 4. VISUALIZZAZIONE (IL CUORE DELLA MODIFICA) ────────────────────────────
layout(matrix(c(2, 4, 1, 3), 2, 2, byrow = TRUE), widths = c(4, 1.5), heights = c(1.5, 4))

# --- Pannello 1: Congiunta ---
par(mar = c(4, 4, 1, 1))
# Calcolo densità congiunta vera per contour
B_o <- 0.5 * (sum((Y_obs-mean(Y_obs))^2) + 6 + n_obs*mean(Y_obs)^2/(n_obs+1))
Z_true <- outer(t1g, t2g, function(a, b) {
  dinvgamma(b, n_obs/2+4, B_o) * dnorm(a, n_obs*mean(Y_obs)/(n_obs+1), sqrt(b/(n_obs+1)))
})

# Sfondo con DRF
kde_drf <- kde2d(sample(th1, 5000, prob=w_drf, replace=T), 
                 sample(th2, 5000, prob=w_drf, replace=T), n=50, lims=c(t1_lim, t2_lim))
image(kde_drf, col = hcl.colors(64, "YlGnBu", rev=T), xlab=expression(theta[1]), ylab=expression(theta[2]))

# MODIFICA: Contorni allargati
# Calcoliamo livelli che coprano il 90% della massa per vedere i cerchi più grandi
levels_vec <- quantile(Z_true, probs = seq(0.5, 0.98, length.out = 5))
contour(t1g, t2g, Z_true, add=T, col="red", lwd=1.5, levels=levels_vec)
points(th1_true, th2_true, pch=4, col="black", lwd=2) # Punto valore vero

# --- Pannello 2: Marginale th1 (In alto) ---
par(mar = c(0, 4, 2, 1))
plot(d1_drf_dens, col="blue", lwd=2, main="", xaxt="n", xlab="", xlim=t1_lim, ylim=c(0, max(d1_drf_dens$y)*1.3))
lines(d1_rf_dens, col="green3", lwd=2)
# La densità vera (approssimata qui per velocità di codice)
lines(t1g, dinvgamma(1,4,3)*dnorm(t1g, th1_true, 0.2), col="red", lwd=2, lty=2) 
polygon(d1_drf_dens, col=adjustcolor("blue", 0.1), border=NA)

# --- Pannello 3: Marginale th2 (A destra) ---
par(mar = c(4, 0, 1, 2))
plot(d2_drf_dens$y, d2_drf_dens$x, type="l", col="blue", lwd=2, yaxt="n", xlab="Density", ylim=t2_lim, xlim=c(0, max(d2_drf_dens$y)*1.3))
lines(d2_rf_dens$y, d2_rf_dens$x, col="green3", lwd=2)
polygon(c(d2_drf_dens$y, rep(0, length(d2_drf_dens$y))), c(d2_drf_dens$x, rev(d2_drf_dens$x)), col=adjustcolor("blue", 0.1), border=NA)

# --- Pannello 4: Legenda ---
par(mar = c(0, 0, 0, 0))
plot.new()
legend("center", legend=c("True", "ABC-RF", "ABC-DRF"), col=c("red", "green3", "blue"), lwd=2, bty="n")