#!/usr/bin/env Rscript

# Rebuild data/generation/synthetic_features.csv from the expanded manual data.
# The saved version was made with R 4.3.3 and synthpop 1.9.2.

suppressPackageStartupMessages(library(synthpop))

args <- commandArgs(trailingOnly = TRUE)
input <- if (length(args) >= 1) args[[1]] else "data/generation/combined_domains_expanded.csv"
output_stem <- if (length(args) >= 2) args[[2]] else "synthetic_features"
seed <- if (length(args) >= 3) as.integer(args[[3]]) else 141727361L
input <- normalizePath(input, mustWork = TRUE)
output_dir <- dirname(output_stem)
output_name <- basename(output_stem)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_dir <- normalizePath(output_dir, mustWork = TRUE)

columns <- c(
  paste0("NER_Tags_", 0:6),
  paste0("POS_Tags_", 0:6),
  "brand_name", "human_name", "abbr", "location_name", "compound",
  "misspelled", "eTLD"
)
visit_sequence <- columns

data <- read.csv(input, header = TRUE)
selected <- data[, names(data) %in% columns]

set.seed(seed)
synthetic <- syn(
  selected,
  minnumlevels = 2,
  visit.sequence = visit_sequence,
  maxfaclevels = 140,
  k = 100000,
  seed = seed
)

previous_directory <- getwd()
setwd(output_dir)
write.syn(synthetic, filename = output_name, filetype = "csv")
setwd(previous_directory)
