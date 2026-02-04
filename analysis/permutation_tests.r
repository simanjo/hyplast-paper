#%%
if (!require(reticulate)) {
  install.packages("reticulate")
  library(reticulate)
}
if (!require(coin)) {
  install.packages("coin")
  library(coin)
}
use_virtualenv("env/", required = TRUE)
source_python("analysis/analysis_utils.py")

#%%
med_helper <- function(scores) {
  scores <- unlist(scores)
  median(scores)
}

range_helper <- function(scores) {
  scores <- unlist(scores)
  max(scores) - min(scores)
}

custom_snr <- function(median, range) {
  1 / (-(median * range))
}

preprocess_results <- function(results) {
  results$salinity <- sapply(
    sapply(results$setting, (\(x) x$salinity)), factor
  )
  results$crange <- sapply(results$setting, (\(x) x$conc_range))
  results$temp <- sapply(results$setting, (\(x) x$temperature))
  results$sorbent <- sapply(
    sapply(results$setting, (\(x) x$sorbent)), factor
  )
  results$model <- sapply(results$model, factor)
  results$fingerprint <- sapply(results$embedding, factor)
  results$score.median <- sapply(results$score, med_helper)
  results$score.range <- sapply(results$score, range_helper)
  results <- na.omit(results)
  results <- subset(results, !sapply(results$score, is.null))
  results$stat <- mapply(custom_snr, results$score.median, results$score.range)
  results$smf_inter <- with(
    results, interaction(sorbent, model, fingerprint), drop = TRUE
  )
  results$mf_inter <- with(
    results, interaction(model, fingerprint), drop = TRUE
  )
  results$stc_inter <- with(
    results, interaction(salinity, temp, crange), drop = TRUE
  )
  results
}

all_results <- preprocess_results(
  load_results(
    "models/", flatten = TRUE, r2_cutoff = 0.85
  )
)

all_results_kf <- all_results[all_results$target == "kf", ]
all_results_n <- all_results[all_results$target == "n", ]

#%%
print("Results for stat ~ model + sorbent + fingerprint | stc_inter")
it <- independence_test(
  stat ~ model + sorbent + fingerprint | stc_inter,
  data=all_results_kf,
  distribution = approximate(nresample = 100000),
  alternative = "two.sided"
)
print("Results for kf")
print(it)
print(pvalue(it, method = "step-down"))

it <- independence_test(
  stat ~ model + sorbent + fingerprint | stc_inter,
  data = all_results_n,
  distribution = approximate(nresample = 100000),
  alternative = "two.sided"
)
print("Results for n")
print(it)
print(pvalue(it, method="step-down"))

#%%
print("Results for stat ~ sorbent + fingerprint | stc_inter for the KRR models only")
it <- independence_test(
  stat ~ sorbent + fingerprint | stc_inter,
  data=all_results_kf[all_results_kf$model == "krr",],
  distribution = approximate(nresample = 100000),
  alternative = "two.sided"
)
print("Results for kf")
print(it)
print(pvalue(it, method = "step-down"))

it <- independence_test(
  stat ~ sorbent + fingerprint | stc_inter,
  data = all_results_n[all_results_kf$model == "krr",],
  distribution = approximate(nresample = 100000),
  alternative = "two.sided"
)
print("Results for n")
print(it)
print(pvalue(it, method="step-down"))

#%%
print("Results for stat ~ sorbent + fingerprint | stc_inter for the RF models only")
it <- independence_test(
  stat ~ sorbent + fingerprint | stc_inter,
  data=all_results_kf[all_results_kf$model == "rf",],
  distribution = approximate(nresample = 100000),
  alternative = "two.sided"
)
print("Results for kf")
print(it)
print(pvalue(it, method = "step-down"))

it <- independence_test(
  stat ~ sorbent + fingerprint | stc_inter,
  data = all_results_n[all_results_kf$model == "rf",],
  distribution = approximate(nresample = 100000),
  alternative = "two.sided"
)
print("Results for n")
print(it)
print(pvalue(it, method="step-down"))

#%%
print("Results for stat ~ salinity + temp + crange | mf_inter")
it <- independence_test(
  stat ~ salinity + temp + crange | mf_inter,
  data = all_results_kf[all_results_kf$model != "pls",],
  distribution = approximate(nresample = 100000),
  alternative = "two.sided"
)
print("Results for kf")
print(it)
print(pvalue(it, method="step-down"))

it <- independence_test(
  stat ~ salinity + temp + crange | mf_inter,
  data = all_results_kf[all_results_kf$model != "pls",],
  distribution = approximate(nresample = 100000),
  alternative = "two.sided"
)
print("Results for n")
print(it)
print(pvalue(it, method="step-down"))


#%%
print("Results for stat ~ salinity + temp + crange | fingerprint for the KRR models only")
it <- independence_test(
  stat ~ salinity + temp + crange | fingerprint,
  data = all_results_kf[all_results_kf$model == "krr",],
  distribution = approximate(nresample = 100000),
  alternative = "two.sided"
)
print("Results for kf")
print(it)
print(pvalue(it, method="step-down"))

it <- independence_test(
  stat ~ salinity + temp + crange | fingerprint,
  data = all_results_kf[all_results_kf$model == "krr",],
  distribution = approximate(nresample = 100000),
  alternative = "two.sided"
)
print("Results for n")
print(it)
print(pvalue(it, method="step-down"))

#%%
print("Results for stat ~ salinity + temp + crange | fingerprint for the RF models only")
it <- independence_test(
  stat ~ salinity + temp + crange | fingerprint,
  data = all_results_kf[all_results_kf$model == "rf",],
  distribution = approximate(nresample = 100000),
  alternative = "two.sided"
)
print("Results for kf")
print(it)
print(pvalue(it, method="step-down"))

it <- independence_test(
  stat ~ salinity + temp + crange | fingerprint,
  data = all_results_kf[all_results_kf$model == "rf",],
  distribution = approximate(nresample = 100000),
  alternative = "two.sided"
)
print("Results for n")
print(it)
print(pvalue(it, method="step-down"))
