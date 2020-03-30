import sys
sys.path.append("../scripts")
import run_attack_args_helper


CLIENT_OPTIONS = {
    "models": {
        "bert-ag-news":                     "BERT AG News",
        "bert-imdb":                        "BERT IMDB Sentiment",
        "bert-mr":                          "BERT Movie Review Sentiment",
        "bert-yelp-sentiment":              "BERT Yelp Sentiment",
        "cnn-ag-news":                      "CNN AG News",
        "cnn-imdb":                         "CNN IMDB Sentiment",
        "cnn-mr":                           "CNN Movie Review Sentiment",
        "cnn-yelp-sentiment":               "CNN Yelp Sentiment",
        "lstm-ag-news":                     "LSTM AG News",
        "lstm-imdb":                        "LSTM IMDB Sentiment",
        "lstm-mr":                          "LSTM Movie Review Sentiment",
        "lstm-yelp-sentiment":              "LSTM Yelp Sentiment",
    },

    "attacks": {
        "beam-search":                      "Beam Search",
        "greedy-word":                      "Greedy Word",
        "greedy-word-wir":                  "Greedy Word with Importance Ranking",
        "ga-word":                          "Genetic Algorithm",
    },

    "goals": {
        "untargeted-classification":        "Untargeted",
        "targeted-classification":          "Targeted",
    },

    "transformations": {
        "word-swap-embedding":              "Word Swap - Embedding",
        "word-swap-homoglyph":              "Word Swap - Homoglyph",
        "word-swap-neighboring-char-swap":  "Word Swap - Neighboring Character Swap",
    },

    "recipes": {
        "none":                             "None",
        "deepwordbug":                      "Gao2018 - DeepWordBug",
        "alzantot":                         "Alzantot 2018 - Genetic Algorithm",
        "alz-adjusted":                     "Alzantot 2018 - Genetic Algorithm Adjusted",
        "textfooler":                       "Jin 2019 - TextFooler",
        "tf-adjusted":                      "Jin 2019 - TextFooler Adjusted"
    }
}

MODEL_CLASS_NAMES = run_attack_args_helper.MODEL_CLASS_NAMES
ATTACK_CLASS_NAMES = run_attack_args_helper.ATTACK_CLASS_NAMES
GOAL_FUNCTION_CLASS_NAMES = run_attack_args_helper.GOAL_FUNCTION_CLASS_NAMES
TRANSFORMATION_CLASS_NAMES = run_attack_args_helper.TRANSFORMATION_CLASS_NAMES
RECIPE_NAMES = run_attack_args_helper.RECIPE_NAMES


