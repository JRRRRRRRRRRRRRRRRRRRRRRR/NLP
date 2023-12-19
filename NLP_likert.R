# Assuming cdat is your dataframe
library(likert)
file_path <- "NLP_data.csv"
dat <- read.csv(file_path)

cdat <- dat[, -c(1, 2, 4, 6, 8, 10, 11)]      # remove the columns that contain answers to open questions
cdat <- cdat[-c(7, 11),]                      # remove answers from users that did not answer all questions

# Rename the columns
colnames(cdat) <- c("Embeddings", "Sentiment", "Topics", "Combination")

# Function to preprocess Likert data with translation
preprocess_likert_column <- function(column) {
  # Extract numeric response codes
  response_codes <- as.numeric(gsub("[^0-9]", "", column))
  
  # Set up factor with ordered levels
  likert_levels_dutch <- c("Heel slecht", "Slecht", "Niet goed/ Niet slecht", "Goed", "Heel goed")
  
  # Translation from Dutch to English
  likert_levels_english <- c("Very bad", "Bad", "Not good/Not bad", "Good", "Very good")
  
  likert_factor <- factor(response_codes, levels = 1:5, labels = likert_levels_english, ordered = TRUE)
  
  return(likert_factor)
}

# Apply the preprocessing function to each Likert column
likert_columns <- lapply(cdat, preprocess_likert_column)

# Combine the processed columns into a new dataframe
likert_data <- as.data.frame(likert_columns)

# New data
ndat <- c(3,3,1,2,4,1,3,1,4,2,3,2,1,4,4,2,3,2,1,2,2,3,2)

# Create a data frame with the new data
ndat_df <- data.frame(Random = ndat)

# Preprocess the "Random" column
processed_random <- preprocess_likert_column(ndat_df$Random)

# Add the processed "Random" column to the existing likert_data data frame
likert_data$Random <- processed_random

# Generate and plot the Likert object
likert_object <- likert(likert_data)
plot(likert_object)

