source("ToyModel.R")
library(MASS)

hh <- function(x, breaks = 30, col_hist, col_line, main_title,yl = "Prob/Pesi", xl = "Theta") {
  h <- hist(x, breaks = breaks, plot = FALSE)
  base <- diff(h$breaks)[1] #Calcolo la larghezza dei bin per trasformare Dens <- Pesi
  
  # (counts / total = probabilità)
  h$density <- h$counts / length(x) 
  plot(h, freq = FALSE, col = col_hist, border = "black", 
       main = main_title, ylab = yl, xlab = xl)
  
  # curva di densità
  d <- density(x)
  d$y <- d$y * base
  lines(d, col = col_line, lwd = 2.5)
}

#1) CASO UNIDIMENSIONALE
par(mfrow = c(1, 2))
#TH1
hh(th1, col_hist = "darkolivegreen1", col_line = "blue3", main_title = "Senza Resampling",xl=expression(theta[1]))
abline(v = th1_vero, col = "red", lwd = 4, lty=3)

hh(th1_drf,col_hist = "darkolivegreen1", col_line = "blue3", main_title = "Con Resampling",xl=expression(theta[1]))
abline(v = th1_vero, col = "red", lwd = 4, lty=3)
#TH2

hh(th2,breaks = 15, col_hist = "darkolivegreen1", col_line = "blue3", main_title = "Senza Resampling",xl=expression(theta[2]))
abline(v = th2_vero, col = "red", lwd = 4, lty=3)

hh(th2_drf,col_hist = "darkolivegreen1", col_line = "blue3", main_title = "Con Resampling",xl=expression(theta[2]))
abline(v = th2_vero, col = "red", lwd = 4, lty=3)

#2) CASO HEATMAP
par(mfrow=c(1,2))
# Recupero i livelli per il contour dalla tua mappa_joint originale
livelli <- seq(min(mappa_joint), max(mappa_joint), length.out = 9)
# ==========================================
# a) HEATMAP A QUADRATONI (No Resample)
# ==========================================
n_breaks <- 70
breaks_th1 <- seq(-1, 3, length.out = n_breaks)
breaks_th2 <- seq(0.01, 4, length.out = n_breaks)

bin_th1 <- cut(th1, breaks = breaks_th1, include.lowest = TRUE)
bin_th2 <- cut(th2, breaks = breaks_th2, include.lowest = TRUE)

matrice_pesi <- tapply(pesi, list(bin_th1, bin_th2), sum)
matrice_pesi[is.na(matrice_pesi)] <- 0 

centri_th1 <- breaks_th1[-1] - diff(breaks_th1)/2
centri_th2 <- breaks_th2[-1] - diff(breaks_th2)/2

# Disegno la prima heatmap
image(x = centri_th1, y = centri_th2, z = matrice_pesi,
      col = hcl.colors(8, "plasma"),
      xlim = c(-1, 3), ylim = c(0.01, 4),
      xlab = "Theta 1", ylab = "Theta 2",
      main = "Heatmap Non-Resampled")

# Aggiungo la true posterior (Curve di livello bianche)
contour(asse1_joint, asse2_joint, mappa_joint, add = TRUE, col = "white", lty = 6, lwd = 1, levels = livelli)
points(th1_vero, th2_vero, lwd = 3, pch = 4, cex = 2, col = "green")
# Legenda finale
legend("topleft",
       legend = c("True joint post", "True (th1, th2)"),
       text.col = "white",
       col    = c("white", "green"),
       lty    = c(6, NA),
       pch    = c(NA, 4),
       lwd    = c(1, 3),
       bty    = "n")


# ==========================================
# 2) HEATMAP KDE LISCIATA (Ricampionati)
# ==========================================
kde_joint <- kde2d(
  x = th1_drf,
  y = th2_drf,
  n = 200,
  lims = c(-1, 3, 0.01, 4)
)

# Disegno la seconda heatmap
image(kde_joint,
      col = hcl.colors(8, "plasma"),
      xlim = c(-1, 3), ylim = c(0.01, 4),
      xlab = "Theta 1", ylab = "Theta 2",
      main = "Heatmap Resampled")

# Aggiungo la true posterior (Curve di livello bianche)
contour(asse1_joint, asse2_joint, mappa_joint, add = TRUE, col = "white", lty = 6, lwd = 1, levels = livelli)
points(th1_vero, th2_vero, lwd = 3, pch = 4, cex = 2, col = "green")

# Legenda finale
legend("topright",
       legend = c("True joint post", "True (th1, th2)"),
       text.col = "white",
       col    = c("white", "green"),
       lty    = c(6, NA),
       pch    = c(NA, 4),
       lwd    = c(1, 3),
       bty    = "n")


##########################
cat("\nTraining Univariate DRF per Theta 1...\n")
set.seed(1) 
drf_th1 <- drf(
  X = Xref,
  Y = th1,  # Target solo th 1
  num.trees = n_trees,
  num.threads = n_cores
)
pesi_1 <- predict(drf_th1, newdata = y_vero_matx)$weights[1,]

th1sorted <- sort(th1)
pesi1sort <- pesi_1[order(th1)]
plot(x=th1sorted,y=pesi1sort,type="l")

cat("Training Univariate DRF per Theta 2...\n")
set.seed(1)
drf_th2 <- drf(
  X = Xref,
  Y = th2,  # Target solo th 2
  num.trees = n_trees,
  num.threads = n_cores
)
pesi_2 <- predict(drf_th2, newdata = y_vero_matx)$weights[1,]

cat("Calcolo i pesi combinati Omega...\n")
# w = (w1 + w2 - w1*w2) - abs(w1 - w2)
omega <- (pesi_1 + pesi_2 - pesi_1 * pesi_2) - abs(pesi_1 - pesi_2)

cat("Generazione Heatmap 2D con pesi Omega combinati...\n")
cat("Ricampionamento e calcolo KDE per i pesi combinati Omega...\n")

# resampl
sampling_omega <- sample(n_sim, size = 10000, replace = TRUE, prob = omega)
th1_omega <- th1[sampling_omega]
th2_omega <- th2[sampling_omega]

# kde su 
kde_omega <- kde2d(
  x = th1_omega,
  y = th2_omega,
  n = 200,
  lims = c(-1, 3, 0.01, 4) # Stessi limiti della tua kde originale
)


par(mfrow = c(1, 2))

#preso da tm
image(kde_joint,
      col = hcl.colors(16, "plasma"),
      xlim = c(-1, 3), ylim = c(0.01, 4),
      xlab = expression(theta[1]), 
      ylab = expression(theta[2]),
      main = "DRF joint posterior")
contour(asse1_joint, asse2_joint, mappa_joint, add = TRUE, col = "white", lty = 6, lwd = 1, levels = livelli)
points(th1_vero, th2_vero, lwd = 3, pch = 4, cex = 2, col = "green")

#kde su pesi penalita
image(kde_omega,
      col = hcl.colors(16, "plasma"),
      xlim = c(-1, 3), ylim = c(0.01, 4),
      xlab = expression(theta[1]), 
      ylab = expression(theta[2]),
      main = "Pesi con Penalita")
contour(asse1_joint, asse2_joint, mappa_joint, add = TRUE, col = "white", lty = 6, lwd = 1, levels = livelli)
points(th1_vero, th2_vero, lwd = 3, pch = 4, cex = 2, col = "green")







matrice_pesi <- tapply(pesi, list(bin_th1, bin_th2), sum)
matrice_pesi[is.na(matrice_pesi)] <- 0 

matrice_omega <- tapply(omega, list(bin_th1, bin_th2), sum)
matrice_omega[is.na(matrice_omega)] <- 0 


par(mfrow = c(1, 2))
#quadratoni
image(x = centri_th1, y = centri_th2, z = as.matrix(matrice_pesi),
      col = hcl.colors(16, "plasma"),
      xlim = c(-1, 3), ylim = c(0.01, 4),
      xlab = expression(theta[1]), 
      ylab = expression(theta[2]),
      main = "DRF Joint")

contour(asse1_joint, asse2_joint, mappa_joint, add = TRUE, col = "white", lty = 6, lwd = 1, levels = livelli)
points(th1_vero, th2_vero, lwd = 3, pch = 4, cex = 2, col = "green")

#quadratoni
image(x = centri_th1, y = centri_th2, z = as.matrix(matrice_omega),
      col = hcl.colors(16, "plasma"),
      xlim = c(-1, 3), ylim = c(0.01, 4),
      xlab = expression(theta[1]), 
      ylab = expression(theta[2]),
      main = "Pesi con Penalita")


contour(asse1_joint, asse2_joint, mappa_joint, add = TRUE, col = "white", lty = 6, lwd = 1, levels = livelli)
points(th1_vero, th2_vero, lwd = 3, pch = 4, cex = 2, col = "green")

legend("topright",
       legend = c("True joint post", "True (th1, th2)"),
       text.col = "white",
       col    = c("white", "green"),
       lty    = c(6, NA),
       pch    = c(NA, 4),
       lwd    = c(1, 3),
       bty    = "n")


