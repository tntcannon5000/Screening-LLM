# Load required libraries
library(tidyverse)

# Create the dataset
data <- tibble(
  interview = c("1234", "1234", "1235", "1235"),
  scraper = c("w/o scraper", "w scraper", "w/o scraper", "w scraper"),
  result = c("fail", "pass", "fail", "pass")
)

# Create the plot
ggplot(data, aes(x = interview, y = result, fill = scraper)) +
  geom_tile(color = "white", size = 1) +
  scale_fill_manual(values = c("w/o scraper" = "red", "w scraper" = "green")) +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    axis.text = element_text(color = "black"),
    axis.title = element_text(color = "black"),
    legend.text = element_text(color = "black"),
    legend.title = element_text(color = "black"),
    plot.title = element_text(color = "black"),
    axis.text.y = element_text(angle = 0)
  ) +
  labs(title = "Interview Results with and without Scraper",
       x = "Interview ID",
       y = "Result",
       fill = "Scraper Usage") +
  theme(axis.text.y = element_text(angle = 0))

# Save the plot (optional)
ggsave("plot/scrapper_summary_plot.png", width = 8, height = 6)