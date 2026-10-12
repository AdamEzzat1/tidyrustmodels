# Export the public reference datasets from R packages to CSV with a fixed row order.
# Run from the repository root:
#   Rscript reference/tidymodels-r/export_datasets.R
#
# Missing values are written as empty strings. Row order is the package order.

.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages(suppressWarnings({
  library(modeldata)
  library(jsonlite)
}))
Sys.setlocale("LC_COLLATE", "C")

out_dir <- file.path("reference", "fixtures", "datasets")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_ds <- function(df, name) {
  path <- file.path(out_dir, paste0(name, ".csv"))
  write.csv(df, path, row.names = FALSE, na = "", quote = TRUE)
  cat("wrote", path, nrow(df), "rows\n")
}

# iris: binary target versicolor vs other (levels sorted bytewise: other < versicolor)
iris2 <- datasets::iris
iris2$is_versicolor <- ifelse(iris2$Species == "versicolor", "versicolor", "other")
iris2$Species <- NULL
write_ds(iris2, "iris")

# penguins (modeldata): regression target body_mass_g, drop rows with missing outcome
peng <- as.data.frame(modeldata::penguins)
peng_reg <- peng[!is.na(peng$body_mass_g), ]
peng_reg$year <- NULL
write_ds(peng_reg, "penguins")

# penguins_sex: classification target sex (female < male), drop rows with missing outcome
peng_sex <- peng[!is.na(peng$sex), ]
peng_sex$year <- NULL
write_ds(peng_sex, "penguins_sex")

# ames (modeldata): subset of predictors used by the ames_reg task plus outcome
ames <- as.data.frame(modeldata::ames)
ames_sub <- ames[, c("Sale_Price", "Gr_Liv_Area", "Year_Built", "Total_Bsmt_SF", "Bldg_Type", "Neighborhood")]
ames_sub$Bldg_Type <- as.character(ames_sub$Bldg_Type)
ames_sub$Neighborhood <- as.character(ames_sub$Neighborhood)
write_ds(ames_sub, "ames")

prov <- list(
  exported_on = format(Sys.time(), "%Y-%m-%d"),
  r_version = R.version.string,
  modeldata_version = as.character(packageVersion("modeldata")),
  datasets = list(
    iris = list(source = "R datasets::iris (Fisher/Anderson), public domain", rows = nrow(iris2)),
    penguins = list(source = "modeldata::penguins (Palmer penguins, CC0), rows with non-missing body_mass_g", rows = nrow(peng_reg)),
    penguins_sex = list(source = "modeldata::penguins (Palmer penguins, CC0), rows with non-missing sex", rows = nrow(peng_sex)),
    ames = list(source = "modeldata::ames (De Cock 2011, Ames Housing), 5 predictors + Sale_Price", rows = nrow(ames_sub))
  )
)
write_json(prov, file.path(out_dir, "provenance_r.json"), auto_unbox = TRUE, pretty = TRUE)
cat("wrote provenance_r.json\n")
