# Install necessary packages if you haven't already
# install.packages(c("readxl", "ggplot2"))
# install.packages("stringr")

# Load the packages
library(readxl)
library(ggplot2)
library(stringr)

setwd("D:/Kent/University Of Kent UK/Projects/Disso/Screening-LLM/visualisation")

# Read the data from the Excel file
df <- read_excel("excel/Consistency-Test.xlsx")

# Reshape the data for ggplot2
df_long <- reshape2::melt(df, id.vars = c("Category", "Interview"),
                          measure.vars = c("Pass", "Fail"),
                          variable.name = "Outcome", value.name = "Count")
df_long$Interview <- str_wrap(df_long$Interview, width = 20)
# Create the grouped bar plot
plot <- ggplot(df_long, aes(x = factor(Interview), y = Count, fill = Outcome)) +
  geom_bar(stat = "identity", position = "stack") +
  geom_text(aes(label = Count), position = position_stack(vjust = 0.5), color = "black", size = 3) +
  facet_wrap(~ Category, ncol = 2, scales = "free_y") +
  labs(title = "Pass/Fail Count for Interviews by Category",
       x = "Interview ID",
       y = "Count",
       fill = "Outcome") +
  scale_fill_manual(values = c("Pass" = "green", "Fail" = "red")) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y = element_text(angle = 0, hjust = 1)) +
  scale_y_continuous(breaks = function(x) seq(0, ceiling(max(x)), by = 2),
                     expand = expansion(mult = c(0, 0.1))) # Reduce space at the top

# Print the plot
print(plot)

# Save the plot as a PNG file
ggsave(filename = "plot/consistency_plot.png", plot = plot, path = "D:/Kent/University Of Kent UK/Projects/Disso/Screening-LLM/visualisation", width = 12, height = 8, dpi = 300)