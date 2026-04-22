# Funzione custom per plottare istogramma e linea scalati in probabilità
hh <- function(x, breaks = 50, col_hist, col_line, main_title) {
  # 1. Calcola istogramma
  h <- hist(x, breaks = breaks, plot = FALSE)
  base <- diff(h$breaks)[1]
  
  # 2. Scala e plotta l'istogramma (counts / total = probabilità)
  h$density <- h$counts / length(x) 
  plot(h, freq = FALSE, col = col_hist, border = "white", 
       main = main_title, ylab = "Probabilità", xlab = "Theta")
  
  # 3. Scala e aggiungi la curva di densità
  d <- density(x)
  d$y <- d$y * base
  lines(d, col = col_line, lwd = 2.5)
}

#1) CASO UNIDIMENSIONALE
par(mfrow = c(2, 2))
#TH1

hh(th1, breaks = 50, col_hist = "steelblue", col_line = "darkorange", main_title = "Pesi DRF per th1")
abline(v = th1_vero, col = "green", lwd = 2)

hh(th1_drf, breaks = 50, col_hist = "darkblue", col_line = "salmon", main_title = "Istogramma th1 Ricampionato")
abline(v = th1_vero, col = "green", lwd = 2)

#TH2

hh(th2, breaks = 50, col_hist = "olivedrab", col_line = "darkorange", main_title = "Pesi DRF per th1")
abline(v = th2_vero, col = "cyan", lwd = 2)

hh(th2_drf, breaks = 50, col_hist = "forestgreen", col_line = "salmon", main_title = "Istogramma th1 Ricampionato")
abline(v = th2_vero, col = "cyan", lwd = 2)

#2) CASO HEATMAP
par(mfrow=c(1,2))
# Recupero i livelli per il contour dalla tua mappa_joint originale
livelli <- seq(min(mappa_joint), max(mappa_joint), length.out = 9)

# ==========================================
# 1) HEATMAP A QUADRATONI (Pesi originali DRF)
# ==========================================
n_breaks <- 40 
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
      col = hcl.colors(12, "plasma"),
      xlim = c(-1, 3), ylim = c(0.01, 4),
      xlab = "Theta 1", ylab = "Theta 2",
      main = "2D Hist: Pesi DRF (No Resampling)")

# Aggiungo la true posterior (Curve di livello bianche)
contour(asse1_joint, asse2_joint, mappa_joint, add = TRUE, col = "white", lty = 6, lwd = 1, levels = livelli)
points(th1_vero, th2_vero, lwd = 3, pch = 4, cex = 2, col = "green")


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
      col = hcl.colors(12, "plasma"),
      xlim = c(-1, 3), ylim = c(0.01, 4),
      xlab = "Theta 1", ylab = "Theta 2",
      main = "KDE 2D: Resampled (th1_drf, th2_drf)")

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

