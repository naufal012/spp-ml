#' Wrapper xgb.train dengan dua pilihan custom objective Poisson
#'
#' Fungsi ini menyederhanakan training XGBoost dengan memilih antara objective standar
#' atau objective dengan pembobotan dinamis. Formatnya meniru lgbpp.
#'
#' @param vol Vektor numerik untuk custom loss.
#' @param params List parameter untuk XGBoost (gunakan nama parameter XGBoost seperti 'eta').
#' @param dynamic_weighted Jika TRUE, gunakan objective dengan pembobotan dinamis.
#' @param F_prime Nilai yang hanya digunakan jika dynamic_weighted = TRUE.
#' @param ... Argumen lain yang diteruskan ke `xgb.train` (misalnya, data, nrounds).
#'
#' @return Objek `xgb.Booster` yang sudah dilatih.

xgbpp <- function(vol, params, ..., dynamic_weighted = FALSE, F_prime = 1) {
  
  # 1. --- Pilih Objective Function Berdasarkan Argumen ---
  
  if (dynamic_weighted) {
    # --- Mode: Dynamic Weighted ---
    cat("--- Menggunakan objective function dengan Dynamic Weights (F_prime) ---\n")
    
    make_dw_poisson_obj <- function(vol_vec, F_prime_val, hess_min = 1e-6) {
      force(vol_vec); force(F_prime_val); force(hess_min)
      
      function(preds, dtrain) {
        label  <- getinfo(dtrain, "label") # getinfo() juga standar untuk XGBoost
        
        weight <- 1 / (1 + exp(preds) * F_prime_val)
        weight <- weight / sum(weight[label < 0] * vol_vec[label < 0])
        
        pos <- label > 0
        neg <- !pos
        
        grad <- numeric(length(label)); hess <- numeric(length(label))
        grad[pos] <- -1; grad[neg] <- exp(preds[neg]) * vol_vec[neg]; grad <- weight * grad
        hess[pos] <- 0; hess[neg] <- exp(preds[neg]) * vol_vec[neg]; hess <- pmax(weight * hess, hess_min)
        
        list(grad = grad, hess = hess)
      }
    }
    poisson_obj <- make_dw_poisson_obj(vol, F_prime)
    
  } else {
    # --- Mode: Standar ---
    cat("--- Menggunakan objective function dengan Static Weights ---\n")
    
    make_poisson_objective <- function(vol_vec) {
      function(preds, dtrain) {
        labels <- getinfo(dtrain, "label")
        weights <- getinfo(dtrain, "weight")
        if (is.null(weights)) weights <- rep(1, length(labels))
        
        dummy <- which(labels < 0)
        event <- which(labels > 0)
        
        norm <- if (length(dummy) > 0) sum(weights[dummy] * vol_vec[dummy]) else 1
        if (abs(norm) < 1e-9) norm <- 1
        weights <- weights / norm
        
        grad <- numeric(length(labels)); hess <- numeric(length(labels))
        grad[event] <- -weights[event]; hess[event] <- 1e-6
        if (length(dummy) > 0) {
          mu <- exp(preds[dummy]); gh <- weights[dummy] * mu * vol_vec[dummy]
          grad[dummy] <- gh; hess[dummy] <- pmax(gh, 1e-6)
        }
        
        return(list(grad = grad, hess = hess))
      }
    }
    poisson_obj <- make_poisson_objective(vol)
  }
  
  # --- Evaluasi Metrik (dengan format return yang disesuaikan untuk XGBoost) ---
  make_poisson_metric <- function(vol_vec) {
    .metric <- function(preds, dtrain) {
      labels <- getinfo(dtrain, "label")
      event <- which(labels > 0)
      dummy <- which(labels < 0)
      err <- numeric(length(labels))
      if (length(event) > 0) err[event] <- -preds[event]
      if (length(dummy) > 0) err[dummy] <- exp(preds[dummy]) * vol_vec[dummy]
      
      # PERBEDAAN KUNCI: Format return value untuk feval di XGBoost
      return(list(name = "Error",value = sum(err) / 1e6, higher_better = FALSE))
    }
    return(.metric)
  }
  poisson_eval <- make_poisson_metric(vol)
  
  # 2. --- Penetapan Parameter & Pemanggilan xgb.train ---
  params$objective <- poisson_obj
  
  # PERBEDAAN KUNCI: Memanggil xgb.train dan menggunakan argumen 'feval'
  model <- xgb.train(
    params = params,
    feval = poisson_eval,
    maximize = FALSE,
    ...
  )
  
  return(model)
}