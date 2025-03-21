Analyses of everyday language showing people more often talk about atypical features ("purple carrot") than typical features ("orange carrot") of things. 

To reproduce all analyses in the paper from our data files, download this repo and knit the file `writing/journal_article/journal_article.Rmd`. All reported analyses and plots are executed in this Rmd file from available data files.

To replicate the entire corpus analysis and experiment, do as follows:

Note that the the LDP (Language Development Project - child corpus) is currently not publicly available because of privacy concerns associated with densely sampled conversational data containing private, identifiable information about children. We provide intermediary data files with extracted adjective-noun pairs from the corpus. The files `ldp_cabnc_data.csv` and `ldp_full_corpus_redacted` provide the adjective-noun pairs, usage data, and utterance order data.

For extracting adjective-noun pairs from the CABNC (adult corpus), run `get_cabnc_transcripts.Rmd`. The CABNC can be downloaded here: https://ca.talkbank.org/access/CABNC.html.

For preparing adjective-noun pairs for the experiment by coding them for articles (a/an), run and follow instructions `article_coding.R`. This involves hand-coding of mass vs. count nouns and some ambiguous a/an cases.

For running the experiment, use the repos https://github.com/cbergey/many_typicality_judgments and https://github.com/cbergey/typicality_front. These must be hosted on the web to run the experiment.

For post-experiment data processing to create final data files, run `process_turk_data.Rmd` and `writing/journal_article/data_processing.Rmd` to generate final data files that can be analyzed in `writing/journal_article/journal_article.Rmd`.

Runtime: Repo should take less than 5 minutes to download. Code extracting adjective-noun pairs may be expected to take a few hours to run. Code extracting language model responses may be expected to take a few hours to run. Other analysis code is expected to take less than 15 minutes to run on a typical desktop computer.

Analysis code was run using the following packages and versions:
R version 4.3.1 (2023-06-16), ggpubr_0.6.0, ggh4x_0.3.0 , udpipe_0.8.11, lubridate_1.9.4, forcats_1.0.0, stringr_1.5.1, dplyr_1.1.4, purrr_1.0.4, readr_2.1.5, tidyr_1.3.1, tibble_3.2.1, tidyverse_2.0.0, weights_1.0.4, Hmisc_5.2-2, broom.mixed_0.2.9.6. ggthemes_5.1.0, lmerTest_3.1-3, lme4_1.1-36, Matrix_1.6-1.1, papaja_0.1.3, tinylabels_0.2.4, xtable_1.8-4, tidyboot_0.1.1, scales_1.3.0, ggridges_0.5.6, here_1.0.1, ggplot2_3.5.1, png_0.1-8     

Python version 3.9.1, torch 1.8.1, numpy 1.20.2, pickle5 0.0.11, pandas 1.2.4, scipy 1.6.3, transformers 4.21.2, seaborn 0.11.2, scikit-learn 0.24.2, matplotlib 3.5.0, nltk 3.6.2, openai 0.27.7, gensim 4.0.1, huggingface-hub 0.9.1, pytorch-pretrained-bert 0.6.2
