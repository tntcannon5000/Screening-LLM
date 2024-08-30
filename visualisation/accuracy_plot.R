library(ggplot2)
library(readxl)
library(dplyr)
library(RColorBrewer)

setwd("D:/Kent/University Of Kent UK/Projects/Disso/Screening-LLM/visualisation")

# Load the data from the Excel file
sentiment_data <- read_excel("excel/Accuracy_variability_plot.xlsx",sheet="ManAccRAGDKFail")

# Ensure 'Accuracy' is numeric
sentiment_data$Accuracy <- as.numeric(sentiment_data$Accuracy)

# Create the Question_Accuracy column
sentiment_data <- sentiment_data %>%
  mutate(Question_Accuracy = paste(Questions, Accuracy, sep = " - "))

# Define positive and negative sentiments
positive_sentiments <- tolower(c("Admiration", "Adoration", "Aesthetic Appreciation","Amusement",
                                 "Awe", "Calmness", "Contentment", "Desire", "Determination", 
                                 "Ecstasy", "Enthusiasm", "Entrancement", "Excitement", "Gratitude", 
                                 "Interest", "Joy", "Love", "Nostalgia", "Pride", "Realization", 
                                 "Relief", "Romance", "Satisfaction", "Surprise (positive)", "Triumph",
                                 "Concentration", "Contemplation", "Strong Interest","Calm", "Content"))

negative_sentiments <- tolower(c("Anger", "Annoyance", "Anxiety", "Awkwardness", "Boredom", 
                         "Confusion", "Contempt", "Disappointment", "Disapproval", "Disgust", 
                         "Distress", "Doubt", "Embarrassment", "Empathic Pain", "Envy", 
                         "Fear", "Guilt", "Horror", "Pain", "Sadness", "Sarcasm", "Shame", 
                         "Surprise (negative)", "Sympathy", "Tiredness",
                         "Identity Hate", "Insult", "Obscene", "Severe Toxic", "Threat", "Toxic",
                         "Craving", "Negative Surprise"))

# Create a sentiment type column
sentiment_data <- sentiment_data %>%
  mutate(Sentiment_Type = case_when(
    tolower(Sentiment) %in% positive_sentiments ~ "Positive",
    tolower(Sentiment) %in% negative_sentiments ~ "Negative"
  ))

print(sentiment_data)

# Find the positive sentiments in sentiment_data
positive_sentiments <- intersect(positive_sentiments, sentiment_data$Sentiment)

# Find the negative sentiments in sentiment_data
negative_sentiments <- intersect(negative_sentiments, sentiment_data$Sentiment)

# Create color palettes
n_positive <- length(positive_sentiments)
n_negative <- length(negative_sentiments)
positive_colors <- colorRampPalette(c("lightgreen", "darkgreen"))(n_positive)
negative_colors <- colorRampPalette(c("pink", "darkred"))(n_negative)

# Combine color palettes
sentiment_colors <- c(setNames(positive_colors, positive_sentiments),
                      setNames(negative_colors, negative_sentiments))

# Create the scatter plot
ggplot(sentiment_data, aes(x = Question_Accuracy, y = Accuracy, color = Sentiment, size = Score)) +
  geom_line(aes(group = 1), color = "blue", linewidth = 1) +
  geom_point(aes(color = Sentiment, size = Score*0.5), alpha = 0.7) +
  scale_color_manual(values = sentiment_colors) +
  scale_size_continuous(range = c(3, 20)) +
  labs(title = "Accuracy and Sentiments Across Interview Questions",
       x = "Interview Question",
       y = "Accuracy",
       color = "Sentiment") +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  ) +
  guides(color = guide_legend(override.aes = list(size = 5)))

# Save the plot (optional)
ggsave("plot/ManAccRAGDKFail_Plot.png", width = 12, height = 8)