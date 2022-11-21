if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ranger)) install.packages("ranger", repos = "http://cran.us.r-project.org")
if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(ranger)
library(gam)

dataset <- read_csv("Anime.csv")

# explorative data analysis
head(dataset)
str(dataset)

# find columns with na
names(which(sapply(dataset, anyNA)))

# find columns with "Unknown"
names(which(sapply(dataset, function(x) any(str_detect(x, "^Unknown$")))))

# Find mismatches between Premiered year and Start_Aired year
dataset %>% filter(str_ends(Premiered, "[0-9]{4}")) %>% 
  filter(str_extract(Start_Aired, "[0-9]{4}$") != str_extract(Premiered, "[0-9]{4}$")) %>% 
  select("Title", "Start_Aired", "Premiered")


# drop columns that seem irrelevant
dataset <- dataset %>% select(-c("Synonyms", "Japanese", "English"))

# drop ranked, popularity as it is merely a representation of order for score 
# and members, respectively
dataset <- dataset %>% select(-c("Ranked", "Popularity"))

# drop synopsis as the present text vary a lot in terms of content, length and writing style, and that presumably is mostly due to the fact that the texts are written by lots of different writers, which is unrelated to the anime itself
dataset <- dataset %>% select(-"Synopsis")

# remove end aired as it's redundant with start_aired and Episodes
dataset <- dataset %>% select(-"End_Aired")

# drop attributes that are considered not available for the prediction, 
# excluding the score value that shall be predicted (as it's need for the test set)
dataset <- dataset %>%
  select(-c("Scored_Users", "Members", "Favorites"))

# drop attributes that were deemed of too minor relevance after initial experimenting with models
dataset <- dataset %>% select(-c("Broadcast", "Producers", "Licensors", "Studios", "Rating"))


# filter out anime that are not yet fully released
dataset <- dataset %>% 
  filter(Status == "Finished Airing") %>%
  select(-c("Status"))

# filter out entries that do not have a score, episode count or duration
dataset <- dataset %>% filter(!is.na(Score) & (!is.na(Episodes)) & (!is.na(Duration_Minutes)))

any_month_regex <- "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"

# restructure and unify date info
dataset <- dataset %>% mutate(
  Start_Aired_Year = case_when(
    Start_Aired == "Unknown" ~ "Unknown",
    str_ends(Start_Aired, "[0-9]{4}") ~ str_extract(Start_Aired, "[0-9]{4}$"),
    as.integer(str_extract(Start_Aired, "[0-9]{2}$")) > 22 ~ 
      paste0("19", str_extract(Start_Aired, "[0-9]{2}$")),
    as.integer(str_extract(Start_Aired, "[0-9]{2}$")) <= 22 ~ 
      paste0("20", str_extract(Start_Aired, "[0-9]{2}$"))
  ),
  Start_Aired_Month = ifelse(
    str_detect(Start_Aired, any_month_regex), 
    str_extract(Start_Aired, any_month_regex),
    "Unknown"
  ),
  Start_Aired_Day = ifelse(str_detect(Start_Aired, "[0-9]{1,2},"), 
                           str_extract(Start_Aired, "[0-9]{1,2}(?=,)"), 
                           "Unknown")
) %>% select(-Start_Aired, -Start_Aired_Day)

dataset <- dataset %>% mutate(
  Start_Aired_Season = ifelse(Premiered == "Unknown", 
                              Premiered, 
                              str_extract(Premiered, "Spring|Summer|Fall|Winter"))
) %>% select(-Premiered)

# drop a few rows that do not have a Start_Aired_Year
dataset <- dataset %>% filter(Start_Aired_Year != "Unknown")


# expand comma-separated values
dataset <- dataset %>% mutate(
  Demographics = str_split(Demographics, ", "),
  Genres = str_split(Genres, ", "),
  Themes = str_split(Themes, ", ")
) %>% unnest(Demographics) %>%
  unnest(Genres) %>%
  unnest(Themes)


# remove whitespaces from strings
dataset <- dataset %>% mutate_at(
  c("Type", "Source", "Genres", "Themes", "Demographics", "Start_Aired_Month", 
    "Start_Aired_Season"), str_replace_all, " ", "_")

# convert columns to Factor or Integer where applicable
dataset <- dataset %>% mutate_at(
  c("Type", "Source", "Genres", "Themes", "Demographics", "Start_Aired_Month", 
    "Start_Aired_Season"), as.factor)
dataset <- dataset %>% mutate_at(
  c("Episodes", "Duration_Minutes", "Start_Aired_Year"), as.integer)


# Create datasets
validation_set <- dataset %>% filter(as.integer(Start_Aired_Year) >= 2020)
data_for_training <- dataset %>% filter(as.integer(Start_Aired_Year) >= 2020)

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = data_for_training$Score, times = 1, 
                                  p = 0.1, list = FALSE)
training_set <- data_for_training[-test_index,]
test_set <- data_for_training[test_index,]

naive <- mean(training_set$Score)
train_lm <- train(Score ~ Type + Episodes + Start_Aired_Year + 
                    Start_Aired_Season + Source + Genres + Themes + 
                    Demographics + Duration_Minutes, 
                  method = "lm", data = training_set)
train_rf <- train(Score ~ Type + Episodes + Source + Demographics, 
                  method = "ranger", data = training_set, num.trees = 50)
train_gl <- train(Score ~ Type + Episodes + Source + Demographics + 
                    Duration_Minutes, 
                  method = "gamLoess", data = training_set)


calculate_RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

names = c("naive", "linear regression", "random forest", "loess")
test_results = c(calculate_RMSE(test_set$Score, naive), calculate_RMSE(test_set$Score, predict(train_lm, test_set)), calculate_RMSE(test_set$Score, predict(train_rf, test_set)), calculate_RMSE(test_set$Score, predict(train_gl, test_set)))

validation_results = c(calculate_RMSE(validation_set$Score, naive), calculate_RMSE(validation_set$Score, predict(train_lm, validation_set)), calculate_RMSE(validation_set$Score, predict(train_rf, validation_set)), calculate_RMSE(validation_set$Score, predict(train_gl, validation_set)))

names(test_results) <- names
test_results

names(validation_results) <- names
validation_results