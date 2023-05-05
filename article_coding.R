library(tidyverse)
library(here)

### This script is for designating appropriate indefinite articles for nouns and adjectives in a list of noun-adj pairs.

### Generally, words that start with consonants are preceded by the indefinite article "a" and words that start
### with vowels are preceded by the indefinite article "an" (e.g., "a cat", "an apple").
### However, there are exceptions (e.g., "an honest person", "a union", "a once-in-a-lifetime event").

# We'll hand-check likely exceptions (words beginning with "o","u", and "h") and add those exceptions to 
# a list, and then code the rest of the words according to the vowel--an consonant--a rule.
adj_noun_pairs <- read_csv(here("data/all_cabnc_ldpParentandKid_conc_unique_pairs.csv")) %>%
  select(adj_token, noun_token)
vowels <- c("a","e","i") # "o" and "u" are sometimes exceptions to the vowel -> "an" rule
likely_exceptions <- c("o","u","h") # exceptions to article rules 

adj_exceptions <- adj_noun_pairs %>% distinct(adj_token) %>% filter(substr(adj_token,1,1) %in% likely_exceptions)
noun_exceptions <- adj_noun_pairs %>% distinct(noun_token) %>% filter(substr(noun_token,1,1) %in% likely_exceptions)

#write_csv(adj_exceptions, here("data/adj_exceptions.csv"))
#write_csv(noun_exceptions, here("data/noun_exceptions.csv"))

# Here, code the articles of the exceptions in each csv, in a separate column called "noun_article" or "adj_article".
# Then we'll read these codes back in.

coded_adj_exceptions <- read_csv(here("data/adj_exceptions.csv")) 
coded_noun_exceptions <- read_csv(here("data/noun_exceptions.csv"))

# We already coded whether the nouns are mass vs. count nouns ("some rice" vs. "a house"). 
# We also made plural nouns singular, when applicable (unless they are conventionally plural, e.g., "scissors".)
# Reading in noun data with singular forms and mass vs. count...
nouns <- read_csv(here("data/all_cabnc_ldpParentandKid_conc_unique_NOUNs_coded.csv")) %>% select(noun_token, noun, mass)

coded_pairs <- adj_noun_pairs %>%
  left_join(nouns) %>%
  left_join(coded_adj_exceptions) %>%
  left_join(coded_noun_exceptions)

coded_pairs <- coded_pairs %>%
  mutate(adj_article = case_when(mass == TRUE ~ "",
                                 !is.na(adj_article) ~ adj_article,
                                 substr(adj_token,1,1) %in% vowels ~ "an",
                                 !(substr(adj_token,1,1) %in% vowels) ~ "a"))

coded_pairs <- coded_pairs %>%
  mutate(article = case_when(mass == TRUE ~ "",
                             !is.na(noun_article) ~ noun_article,
                             substr(noun_token,1,1) %in% vowels ~ "an",
                             !(substr(noun_token,1,1) %in% vowels) ~ "a"))

coded_pairs <- coded_pairs %>%
  rename(adj = adj_token, noun_orig = noun_token) %>%
  mutate(noun = if_else(is.na(noun), noun_orig, noun),
         mass = if_else(is.na(mass), "FALSE", "TRUE"))

clean_pairs <- coded_pairs %>%
  select(adj, noun, adj_article, article, mass) %>%
  distinct(adj, noun, adj_article, article, mass)

write_csv(coded_pairs, here("data/ldp_cabnc_pairs_all_info.csv"))
write_csv(clean_pairs, here("data/ldp_cabnc_pairs_for_turk.csv"))
