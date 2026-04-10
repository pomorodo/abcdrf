# ============================================================
#  ABC e Distributional Random Forests – Toy Model
#  Replica della figura in Dinh, Tavaré & Xiang (2024)
# ============================================================

# abcrf  → implementa ABC con Random Forest (Raynal et al. 2019)
# drf    → implementa Distributional Random Forests (Cevid et al. 2022)
# invgamma → fornisce dinvgamma/rinvgamma per la distribuzione Inverse-Gamma
# MASS   → fornisce kde2d, stimatore KDE bivariato su griglia
library(abcrf)
library(drf)
library(invgamma)
library(MASS)

#set.seed(111)


# ── 1. PARAMETRI DEL MODELLO ─────────────────────────────────────────────────
# Il modello gerarchico è:
#   Y_i | θ  ~ N(θ1, θ2)          (likelihood)
#   θ1  | θ2 ~ N(0, θ2)           (prior su θ1 condizionata a θ2)
#   θ2        ~ IG(alpha0, beta0)  (prior marginale su θ2)
# I valori alpha0=4, beta0=3 sono esplicitamente indicati nel paper (Sezione 3.1).

alpha0 <- 4    # parametro di forma della Inverse-Gamma
beta0  <- 3    # parametro di scala (rate) della Inverse-Gamma
n_obs  <- 500   # numero di osservazioni nel campione "osservato"
# (il paper non specifica n; 50 è una scelta ragionevole)


# ── 2. FUNZIONE DELLE STATISTICHE SOMMARIO ───────────────────────────────────
# Il paper usa 61 feature in totale: 11 legate al modello + 50 rumore puro.
# Le 11 statistiche sommario riproducono quelle di Raynal et al. (2019)
# per questo stesso toy model. Sono sufficienti perché catturano la
# struttura di un campione normale (posizione, scala, forma).

sumstats <- function(Y) {
  
  Ybar <- mean(Y)          # media campionaria: statistica sufficiente per θ1
  s    <- sd(Y)            # deviazione standard campionaria (usata sotto per normalizzare)
  S2   <- sum((Y - Ybar)^2) # somma degli scarti quadratici: proporzionale a S² del paper,
  # statistica sufficiente per θ2
  
  qs <- as.numeric(quantile(Y, c(0.1, 0.25, 0.75, 0.9)))
  # quantili empirici al 10°, 25°, 75°, 90° percentile:
  # catturano la forma della distribuzione campionaria meglio della sola varianza
  
  c(Ybar  = Ybar,            # 1. media
    S2    = S2,             # 2. somma scarti quadratici
    varY  = var(Y),         # 3. varianza campionaria (con divisione n-1, diversa da S2/n)
    medY  = median(Y),      # 4. mediana: robusta agli outlier
    madY  = mad(Y),         # 5. MAD (Median Absolute Deviation): misura robusta di scala
    q10   = qs[1],          # 6. 1° decile
    q25   = qs[2],          # 7. 1° quartile
    q75   = qs[3],          # 8. 3° quartile
    q90   = qs[4],          # 9. 9° decile
    skew  = mean((Y - Ybar)^3) / s^3,  # 10. skewness standardizzata (momento terzo)
    kurt  = mean((Y - Ybar)^4) / s^4)  # 11. kurtosi standardizzata (momento quarto)
}


# ── 3. GENERAZIONE DEL REFERENCE SET ─────────────────────────────────────────
# Il paper specifica esattamente N = 10 000 simulazioni dal modello (Sezione 3.1).
# Questo è il "reference set": l'insieme di coppie (parametri, statistiche)
# su cui verrà addestrata la foresta. Più è grande, più precisa è l'inferenza.

N <- 10000

# Passo 1: campionare θ2 dalla sua prior marginale IG(alpha0, beta0).
# rinvgamma usa la parametrizzazione rate (= beta), non scale.
th2 <- rinvgamma(N, shape = alpha0, rate = beta0)

# Passo 2: campionare θ1 dalla sua prior condizionata a θ2.
# θ1 | θ2 ~ N(0, θ2), quindi la deviazione standard è sqrt(θ2).
th1 <- rnorm(N, mean = 0, sd = sqrt(th2))

cat("Generazione reference set...\n")

# Passo 3: per ciascuna delle N simulazioni, generare un campione Y di
# dimensione n_obs e calcolarne le 11 statistiche sommario.
# sapply itera su 1:N; t() traspone perché sapply restituisce colonne.
S_ref <- t(sapply(seq_len(N), function(i)
  sumstats(rnorm(n_obs, mean = th1[i], sd = sqrt(th2[i])))))
# S_ref è una matrice N × 11

# Passo 4: aggiungere 50 colonne di rumore uniforme U(0,1).
# Questo riproduce fedelmente il setup del paper (Sezione 3.1: "50 of which
# were U(0,1) noise") e serve a testare la capacità della foresta di
# ignorare le feature irrilevanti (variable importance).
noise_ref       <- matrix(runif(N * 50), nrow = N, ncol = 50)
colnames(noise_ref) <- paste0("u", 1:50)   # nomi u1, u2, ..., u50

# Passo 5: unire le 11 statistiche reali e le 50 di rumore → 61 feature totali.
X_ref <- cbind(S_ref, noise_ref)
# X_ref è una matrice N × 61: input della foresta


# ── 4. DATI OSSERVATI ─────────────────────────────────────────────────────────
# Generiamo UN singolo dataset "osservato" come se fosse quello reale.
# I valori th1_true e th2_true sono i parametri veri che vogliamo recuperare.

th2_true <- rinvgamma(1, alpha0, beta0)         # un solo valore da IG
th1_true <- rnorm(1, mean = 0, sd = sqrt(th2_true)) # un solo valore da N(0, θ2)
Y_obs    <- rnorm(n_obs, mean = th1_true, sd = sqrt(th2_true))
# Y_obs: campione di n_obs osservazioni "reali"

# Calcoliamo le statistiche sommario del dato osservato + 50 valori di rumore.
# Anche il dato osservato deve avere esattamente la stessa struttura del reference set.
x_obs <- c(sumstats(Y_obs), runif(50))
names(x_obs)[12:61] <- paste0("u", 1:50)   # assegniamo i nomi alle 50 feature rumore

# Convertiamo in data.frame con le stesse colonne di X_ref (richiesto da predict.regAbcrf)
x_obs_df <- as.data.frame(t(x_obs))
colnames(x_obs_df) <- colnames(X_ref)

cat(sprintf("Valori veri:  θ1 = %.3f   θ2 = %.3f\n", th1_true, th2_true))


# ── 5. POSTERIORE ANALITICA ESATTA ───────────────────────────────────────────
# Questo toy model è uno dei pochi casi in cui la posteriore è nota in forma chiusa.
# Le formule vengono dalle equazioni (1) e (2) del paper.

Ybar_o <- mean(Y_obs)                          # media del campione osservato
S2_o   <- sum((Y_obs - Ybar_o)^2)              # somma scarti quadratici del campione osservato

# B è la statistica cerniera che dipende dai dati e compare in entrambe le posteriari.
# Formula dalla Sezione 3 del paper:  B = (1/2)(S² + 6 + n·Ȳ²/(n+1))
B <- 0.5 * (S2_o + 6 + n_obs * Ybar_o^2 / (n_obs + 1))

# Parametri della posteriore di θ2 | Y ~ IG(a_post, b_post)  [equazione (1) del paper]
a_post <- n_obs / 2 + 4   # parametro di forma aggiornato
b_post <- B               # parametro di scala (rate) aggiornato

# Parametri della posteriore marginale di θ1 | Y ~ t_{n+8}(mu1, tau²) [sezione 3]
mu1  <- n_obs * Ybar_o / (n_obs + 1)                  # media della t
tau2 <- 2 * B / ((n_obs + 1) * (n_obs + 8))           # varianza della t
df1  <- n_obs + 8                                      # gradi di libertà


# ── 6. ABC-RF ────────────────────────────────────────────────────────────────
# ABC-RF addestra una foresta di regressione separata per OGNI parametro scalare.
# Non può stimare la posteriore congiunta direttamente: è il limite principale
# rispetto ad ABC-DRF (discusso nella Sezione 2 del paper).

# Costruiamo un data.frame con la colonna del parametro + le 61 feature

# Foresta per θ1: la formula "theta1 ~ ." usa tutte le 61 colonne come predittori.
# ntree = 500 alberi: valore standard, bilanciamento tra accuratezza e velocità.
# paral = FALSE: non usare parallelizzazione (per compatibilità; mettere TRUE se si ha OpenMP)

# Foresta analoga per θ2

# Predizione sul dato osservato: la foresta assegna pesi locali ai punti
# del reference set in base a quante volte co-appaiono nella stessa foglia.
# L'output contiene: $weights (pesi su N punti), $expectation, $med, $variance...

# Estraiamo i pesi e li normalizziamo a somma 1 (potrebbero già esserlo,
# ma la normalizzazione esplicita evita errori numerici).
# normalizzazione

# Stimiamo la densità marginale posteriore come KDE pesata sul reference set.
# bw = "SJ": selettore di banda di Sheather-Jones, più accurato del default "nrd0"
#            per distribuzioni asimmetriche come la Inverse-Gamma.
# n = 512: numero di punti della griglia su cui valutare la densità.
# ── 6. ABC-RF ────────────────────────────────────────────────────────────────
#cat("Fit ABC-RF...\n")
#ref_df <- as.data.frame(X_ref)

#rf_t1 <- regAbcrf(theta1 ~ .,
            #      data  = data.frame(theta1 = th1, ref_df),
            #      ntree = 500, paral = TRUE,
            #      ncores = parallel::detectCores() - 1)

#rf_t2 <- regAbcrf(theta2 ~ .,
              #    data  = data.frame(theta2 = th2, ref_df),
              #    ntree = 500, paral = TRUE,
              #    ncores = parallel::detectCores() - 1)

# Estraiamo i pesi usando la foresta ranger sottostante direttamente
# rf_t1$model.rf è l'oggetto ranger; predict su ranger restituisce
# le foglie per ogni albero → da cui ricaviamo i pesi locali
#get_abcrf_weights <- function(rf_obj, x_obs_df, th_ref) {
  # Predizione foglie: matrice N_ref × ntree
 # leaves_ref <- predict(rf_obj$model.rf,
                  #      data        = ref_df,
                  #      type        = "terminalNodes")$predictions
  # Foglie per l'osservazione target: vettore di lunghezza ntree
 # leaves_obs <- predict(rf_obj$model.rf,
                     #   data        = x_obs_df,
                     #   type        = "terminalNodes")$predictions[1, ]
  
  # Peso = proporzione di alberi in cui ref[i] finisce nella stessa foglia di x_obs
#  w <- colMeans(t(leaves_ref) == leaves_obs)
#  w <- pmax(w, 0)
 # w / sum(w)
#}

#w_rf1 <- get_abcrf_weights(rf_t1, x_obs_df, th1)
#w_rf2 <- get_abcrf_weights(rf_t2, x_obs_df, th2)

#d1_rf <- density(th1, weights = w_rf1, n = 512, bw = "SJ")
#d2_rf <- density(th2, weights = w_rf2, n = 512, bw = "SJ")

#cat("Range d1_rf$x:", range(d1_rf$x), "\n")
#cat("Range d2_rf$x:", range(d2_rf$x), "\n")
# ── 7. ABC-DRF ───────────────────────────────────────────────────────────────
# DRF è una generalizzazione multivariata: stima la distribuzione congiunta
# di (θ1, θ2) in un unico passo, catturando la dipendenza tra i parametri.
# Questo è il contributo chiave del paper (Sezione 2).
cat("Fit ABC-DRF...\n")

# Y è ora una matrice N × 2: la foresta predice distribuzioni su R².
drf_fit <- drf(X         = X_ref,
               Y         = cbind(theta1 = th1, theta2 = th2),
               num.trees = 500)   # stesso numero di alberi di ABC-RF per confronto equo

# Predizione: convertiamo x_obs in matrice 1 × 61 con gli stessi nomi colonna
x_obs_mat           <- matrix(x_obs, nrow = 1)
colnames(x_obs_mat) <- colnames(X_ref)

drf_out <- predict(drf_fit, newdata = x_obs_mat)
# drf_out$weights è una matrice 1 × N: peso di ciascun punto del reference set

# Estraiamo il vettore di pesi per la nostra unica osservazione
w_drf <- drf_out$weights[1, ]

# pmax(w_drf, 0): azzeriamo eventuali pesi negativi (artefatti numerici della foresta)
w_drf <- pmax(w_drf, 0)
w_drf <- w_drf / sum(w_drf)   # normalizzazione a somma 1

# Campionamento dalla posteriore congiunta: peschiamo 5000 indici dal reference set
# proporzionalmente ai pesi DRF. Questi campioni approssimano p(θ1,θ2|Y_obs).
idx_drf <- sample(N, size = 5000, replace = TRUE, prob = w_drf)
th1_drf <- th1[idx_drf]   # campioni posteriori di θ1 secondo DRF
th2_drf <- th2[idx_drf]   # campioni posteriori di θ2 secondo DRF

# Densità marginali DRF: KDE pesata direttamente sui pesi (non sui campioni resampliati)
# Usare i pesi originali è più accurato del resamplig per le marginali 1D.
d1_drf <- density(th1, weights = w_drf, n = 512, bw = "SJ")
d2_drf <- density(th2, weights = w_drf, n = 512, bw = "SJ")


# ── 8. GRIGLIE PER LA DENSITÀ VERA ───────────────────────────────────────────
# Definiamo i limiti del grafico in modo che includano la massa principale
# della posteriore senza tagliare le code.
t1_lim <- c(-0.5, 3.5)    # range per θ1 sull'asse x
t2_lim <- c(0.01, 3.5)    # range per θ2 sull'asse y (0.01 per evitare singolarità in 0)

# Densità congiunta vera p(θ1,θ2|Y) = p(θ1|θ2,Y) × p(θ2|Y)
# scritta come log-somma per stabilità numerica, poi esponenziata
true_joint <- function(t1, t2) {
  exp(
    dinvgamma(t2, shape = a_post, rate = b_post, log = TRUE) +  # log p(θ2|Y)
      dnorm(t1, mean = mu1, sd = sqrt(2 * t2 / (n_obs + 1)), log = TRUE)  # log p(θ1|θ2,Y)
  )
}

# Griglie 1D per valutare la densità congiunta su tutti i punti (g1_i, g2_j)
g1 <- seq(t1_lim[1], t1_lim[2], length.out = 80)
g2 <- seq(t2_lim[1], t2_lim[2], length.out = 80)

# outer() applica true_joint a tutte le combinazioni (g1_i, g2_j):
# risultato è una matrice 80×80 di valori di densità
Z_true <- outer(g1, g2, Vectorize(true_joint))

# Griglie fini per le densità marginali (512 punti per una curva liscia)
t1g <- seq(t1_lim[1], t1_lim[2], length.out = 512)
t2g <- seq(t2_lim[1], t2_lim[2], length.out = 512)

# Marginale vera di θ1: distribuzione t con location mu1, scale sqrt(tau2), df = n+8
# La funzione dt() dà la t standard, quindi dobbiamo scalare manualmente:
# p(θ1|Y) = (1/sqrt(tau2)) × t_{df1}((θ1 - mu1)/sqrt(tau2))
d1_true <- dt((t1g - mu1) / sqrt(tau2), df = df1) / sqrt(tau2)

# Marginale vera di θ2: Inverse-Gamma con parametri aggiornati
d2_true <- dinvgamma(t2g, shape = a_post, rate = b_post)


# ════════════════════════════════════════════════════════════
# 9b.  KDE BIVARIATA AD ALTA RISOLUZIONE
# ════════════════════════════════════════════════════════════
#
#  n=200 invece di 60: griglia 200×200 invece di 60×60.
#  Questo elimina completamente i "quadrati" visibili nella
#  heatmap. Il costo computazionale è trascurabile per N=5000.
#
#  Zoom automatico: calcoliamo i limiti direttamente dai
#  campioni DRF invece di usare limiti fissi t1_lim/t2_lim.
#  In questo modo il pannello è centrato sulla massa della
#  posteriore (zona gialla/verde), dove vivono i contour rossi.

# Percentili al 1% e 99% dei campioni DRF: tagliano le code
# estreme lasciando visibile il 98% della massa posteriore.
t1_zoom <- quantile(th1_drf, c(0.005, 0.995))
t2_zoom <- quantile(th2_drf, c(0.005, 0.995))

# Aggiungiamo un margine del 10% per non tagliare le curve ai bordi
margin1 <- diff(t1_zoom) * 0.10
margin2 <- diff(t2_zoom) * 0.10
t1_zoom <- t1_zoom + c(-margin1, margin1)
t2_zoom <- t2_zoom + c(-margin2, margin2)

# KDE bivariata ad alta risoluzione sulla finestra zoomata.
# n=200: griglia 200×200 → nessun pixel visibile.
# lims: corrisponde esattamente allo zoom → nessuna area vuota.
kde_joint_hd <- kde2d(
  x    = th1_drf,
  y    = th2_drf,
  n    = 200,
  lims = c(t1_zoom, t2_zoom)
)

# Griglia vera congiunta sulla stessa finestra zoomata:
# servono g1_z e g2_z per i contour rossi allineati con la heatmap.
g1_z   <- seq(t1_zoom[1], t1_zoom[2], length.out = 150)
g2_z   <- seq(t2_zoom[1], t2_zoom[2], length.out = 150)
Z_zoom <- outer(g1_z, g2_z, Vectorize(true_joint))

# Griglia fine per le marginali 1D (stessa finestra dello zoom)
t1g <- seq(t1_zoom[1], t1_zoom[2], length.out = 512)
t2g <- seq(t2_zoom[1], t2_zoom[2], length.out = 512)

# Ricalcolo densità vere sulle griglie zoomate
d1_true <- dt((t1g - mu1) / sqrt(tau2), df = df1) / sqrt(tau2)
d2_true <- dinvgamma(t2g, shape = a_post, rate = b_post)


# ════════════════════════════════════════════════════════════
# 10.  FIGURA
# ════════════════════════════════════════════════════════════
#
#    ┌─────────────────────┬───────────┐
#    │  Pannello 2         │ Pannello 4│
#    │  Marginale θ1       │  Legenda  │
#    ├─────────────────────┼───────────┤
#    │  Pannello 1         │ Pannello 3│
#    │  Congiunta 2D       │ Margin.θ2 │
#    │  (heatmap+contour)  │ (ruotata) │
#    └─────────────────────┴───────────┘

layout(
  matrix(c(2, 4, 1, 3), nrow = 2, ncol = 2, byrow = TRUE),
  widths  = c(4, 1.8),
  heights = c(1.8, 4)
)


# ── Pannello 1: Posteriore congiunta zoomata (basso-sinistra) ──
par(mar = c(4, 4, 0, 0))

# image() con kde_joint_hd (200×200): nessun quadrato visibile.
# La palette "YlOrRd" inverte il viridis: giallo=bassa, rosso=alta.
# Usiamo "YlGnBu" che è simile al paper (blu scuro=alta densità).
image(kde_joint_hd,
      col  = hcl.colors(64, "Viridis"),
      # 512 colori: transizione graduale senza banding
      xlim = t1_zoom,
      ylim = t2_zoom,
      xlab = expression(theta[1]),
      ylab = expression(theta[2]))

# Contour rossi più visibili:
#   - lwd=2.5: linee spesse
#   - nlevels=12: più curve per coprire bene la regione zoomata
#   - drawlabels=FALSE: no etichette numeriche, più pulito
contour(g1_z, g2_z, Z_zoom,
        add         = TRUE,
        col         = "red",
        lwd         = 2.5,
        nlevels     = 7,
        drawlabels  = FALSE)

box()


# ── Pannello 2: Marginale θ1 (alto-sinistra) ─────────────────
par(mar = c(0, 4, 2, 0))

# Ricalcolo del limite y sulla finestra zoomata.
# Includiamo d1_rf$y solo nel range t1_zoom per evitare che
# picchi nelle code allarghino inutilmente l'asse y.
in_zoom1 <- d1_rf$x >= t1_zoom[1] & d1_rf$x <= t1_zoom[2]
ylim2 <- c(0, max(d1_true,
                  d1_drf$y[d1_drf$x >= t1_zoom[1] & d1_drf$x <= t1_zoom[2]],
                  d1_rf$y[in_zoom1]) * 1.25)

plot(t1g, d1_true,
     type = "n",
     xlim = t1_zoom,   # stesso zoom del pannello 1
     ylim = ylim2,
     xaxt = "n",
     ylab = "Density", xlab = "")

# Poligono riempito per ABC-DRF
polygon(
  c(d1_drf$x, rev(d1_drf$x)),
  c(d1_drf$y, rep(0, length(d1_drf$y))),
  col    = adjustcolor("steelblue", alpha.f = 0.30),
  border = NA
)

# Le tre curve: ordine di disegno importante.
# Disegniamo verde (ABC-RF) PER ULTIMA così è sempre visibile
# sopra il poligono blu e non viene coperta.
lines(t1g,      d1_true,  col = "red",    lwd = 2.5)  # vera (rosso)
lines(d1_drf$x, d1_drf$y, col = "blue",   lwd = 2)    # ABC-DRF (blu)
#lines(d1_rf$x,  d1_rf$y,  col = "green3", lwd = 2.5)  # ABC-RF (verde) — sopra a tutto


# ── Pannello 3: Marginale θ2 (basso-destra, ruotata 90°) ─────
par(mar = c(4, 0, 0, 2))

in_zoom2 <- d2_rf$x >= t2_zoom[1] & d2_rf$x <= t2_zoom[2]
xlim3 <- c(0, max(d2_true,
                  d2_drf$y[d2_drf$x >= t2_zoom[1] & d2_drf$x <= t2_zoom[2]],
                  d2_rf$y[in_zoom2]) * 1.25)

plot(d2_true, t2g,
     type = "n",
     ylim = t2_zoom,   # stesso zoom del pannello 1
     xlim = xlim3,
     yaxt = "n",
     xlab = "Density", ylab = "")

polygon(
  c(d2_drf$y, rev(rep(0, length(d2_drf$y)))),
  c(d2_drf$x, rev(d2_drf$x)),
  col    = adjustcolor("steelblue", alpha.f = 0.30),
  border = NA
)

# Stesso ordine: verde per ultima → sempre in primo piano
lines(d2_true,  t2g,       col = "red",    lwd = 2.5)  # vera
lines(d2_drf$y, d2_drf$x,  col = "blue",   lwd = 2)    # ABC-DRF
#lines(d2_rf$y,  d2_rf$x,   col = "green3", lwd = 2.5)  # ABC-RF — sopra a tutto


# ── Pannello 4: Legenda (alto-destra) ─────────────────────────
par(mar = c(0, 0, 2, 2))
plot.new()
legend("center",
       legend = c("True posterior", "ABC-DRF"),
       col    = c("red", "blue"),
       lwd    = 2.5,
       bty    = "n",
       cex    = 1.15)

