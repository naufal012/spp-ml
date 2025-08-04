#' Wrapper lgb.train dengan dua pilihan custom objective Poisson
#'
#' Fungsi ini menyederhanakan training dengan memilih antara objective standar
#' (static weights) atau objective dengan pembobotan dinamis (dynamic weights).
#'
#' @param vol Vektor numerik untuk custom loss.
#' @param params List parameter untuk LightGBM.
#' @param dynamic_weighted Logika (TRUE/FALSE). Jika TRUE, gunakan objective
#'   dengan pembobotan dinamis. Default-nya FALSE.
#' @param F_prime Nilai numerik yang hanya digunakan jika dynamic_weighted = TRUE.
#'   Default-nya 1.
#' @param ... Argumen lain yang diteruskan ke `lgb.train` (data, nrounds, dll.).
#'
#' @return Objek `lgb.Booster` yang sudah dilatih.

lgbpp <- function(vol, params, ..., dynamic_weighted = FALSE, F_prime = 1) {
  
  # 1. --- Pilih Objective Function Berdasarkan Argumen ---
  
  if (dynamic_weighted) {
    # --- JIKA dynamic_weighted = TRUE, GUNAKAN RESEP INI ---
    cat("--- Menggunakan objective function dengan Dynamic Weights (F_prime) ---\n")
    
    make_dw_poisson_obj <- function(vol_vec, F_prime_val, hess_min = 1e-6) {
      force(vol_vec); force(F_prime_val); force(hess_min)
      
      function(preds, dtrain) {
        label  <- get_field(dtrain, "label")
        
        # Bobot dihitung secara dinamis di setiap iterasi
        weight <- 1 / (1 + exp(preds) * F_prime_val)
        weight <- weight / sum(weight[label < 0] * vol_vec[label < 0]) # Normalisasi
        
        pos <- label > 0
        neg <- !pos
        
        # Gradien
        grad <- numeric(length(label))
        grad[pos] <- -1
        grad[neg] <- exp(preds[neg]) * vol_vec[neg]
        grad <- weight * grad # Terapkan bobot
        
        # Hessian
        hess <- numeric(length(label))
        hess[pos] <- 0 # Hessian nol untuk titik positif
        hess[neg] <- exp(preds[neg]) * vol_vec[neg]
        hess <- pmax(weight * hess, hess_min)
        
        list(grad = grad, hess = hess)
      }
    }
    poisson_obj <- make_dw_poisson_obj(vol, F_prime)
    
  } else {
    # --- JIKA dynamic_weighted = FALSE (DEFAULT), GUNAKAN RESEP STANDAR ---
    cat("--- Menggunakan objective function dengan Static Weights ---\n")
    
    make_poisson_objective <- function(vol_vec) {
      function(preds, dtrain) {
        labels <- get_field(dtrain, "label")
        weights <- get_field(dtrain, "weight")
        if (is.null(weights)) weights <- rep(1, length(labels))
        
        dummy <- which(labels < 0)
        event <- which(labels > 0)
        
        norm <- if (length(dummy) > 0) sum(weights[dummy] * vol_vec[dummy]) else 1
        if (abs(norm) < 1e-9) norm <- 1
        weights <- weights / norm
        
        grad <- numeric(length(labels))
        hess <- numeric(length(labels))
        
        grad[event] <- -weights[event]
        hess[event] <- 1e-6
        
        if (length(dummy) > 0) {
          mu <- exp(preds[dummy])
          gh <- weights[dummy] * mu * vol_vec[dummy]
          grad[dummy] <- gh
          hess[dummy] <- pmax(gh, 1e-6)
        }
        
        return(list(grad = grad, hess = hess))
      }
    }
    poisson_obj <- make_poisson_objective(vol)
  }
  
  # --- Evaluasi Metrik (tetap sama untuk kedua kasus) ---
  make_poisson_metric <- function(vol_vec) {
    .metric <- function(preds, dtrain) {
      labels <- get_field(dtrain, "label")
      event <- which(labels > 0)
      dummy <- which(labels < 0)
      err <- numeric(length(labels))
      if (length(event) > 0) err[event] <- -preds[event]
      if (length(dummy) > 0) err[dummy] <- exp(preds[dummy]) * vol_vec[dummy]
      return(list(name = "poisson_err", value = sum(err) / 1e6, higher_better = FALSE))
    }
    return(.metric)
  }
  poisson_eval <- make_poisson_metric(vol)
  
  # 2. --- Penetapan Parameter & Pemanggilan lgb.train ---
  params$objective <- poisson_obj
  
  model <- lgb.train(
    params = params,
    eval = poisson_eval,
    ...
  )
  
  return(model)
}