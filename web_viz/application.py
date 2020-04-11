import flask
from flask import Flask
from flask import render_template, request
import textattack
import json
import numpy as np
import pickle
import textattack
import textattack_helper
from fuzzywuzzy import fuzz
from textattack.shared.tokenized_text import TokenizedText
from textattack.attack_results import AttackResult, FailedAttackResult

SENTIMENT_LABEL = {0: "Negative", 1: "Positive"}
AGNEWS_LABEL = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

DEBUG = True

def fuzzy_match_label(text, label_map):
    best_match = 0
    best_ratio = 0.0
    for label in label_map:
        label_str = label_map[label];
        ratio = fuzz.ratio(text, label_str)
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = label
    return best_match

def create_app():

    # This line required for AWS Elastic Beanstalk
    # Specify the template and static folders so that AWS EB can find them!
    # On your dashboard, make sure to set the static to something like: PATH = /static/  DIRECTORY = webapp/static
    # (only then will the CSS work)
    app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')

    @app.route('/')
    @app.route('/index')
    def index():
      return render_template(
        'index.html', 
        title='TextAttack Demo',
        options=textattack_helper.CLIENT_OPTIONS
    )

    # Example of how to add backend model to frontend template.
    @app.route('/about')
    def about():
      return render_template('about.html',title='About TextAttack')
      
    @app.errorhandler(500)
    def internal_error(error):
        return "500 (Internal Error)"

    # Actually run the model
    @app.route('/run_textattack', methods=['POST'])
    def run_textattack():
        print('Generating adverserial sample...')
        data = request.form

        model_class = textattack_helper.MODEL_CLASS_NAMES[data["model"]]
        model = eval(f'{model_class}()')

        if "sentiment" in textattack_helper.CLIENT_OPTIONS["models"][data["model"]].lower():
            label_map = SENTIMENT_LABEL
        elif "AG News" in textattack_helper.CLIENT_OPTIONS["models"][data["model"]]:
            label_map = AGNEWS_LABEL
        else:
            return internal_error(500)

        if data["recipe"] != "none":
            recipe_class = textattack_helper.RECIPE_NAMES[data["recipe"]]
            attack = eval(f"{recipe_class}")(model)
        else:
            transform_class = textattack_helper.TRANSFORMATION_CLASS_NAMES[data["transformation"]]
            transformation = eval(f"{transform_class}")()

            goal_class = textattack_helper.GOAL_FUNCTION_CLASS_NAMES[data["goal"]]
            if data["goal"] == "targeted-classification": 
                target_label_str = data["target_label"]
                target_label = fuzzy_match_label(target_label_str, label_map)
                goal_function = eval(f"{goal_class}")(model, target_class=target_label)
            else:
                 goal_function = eval(f"{goal_class}")(model)

            attack_class = textattack_helper.ATTACK_CLASS_NAMES[data["attack"]]
            attack = eval(f"{attack_class}")(goal_function, transformation)

        tokenized_input = TokenizedText(
                data['input_string'], 
                model.tokenizer
            )

        original_label = goal_function.get_output(tokenized_input)
        attack_result = attack.attack_one(tokenized_input, original_label)

        if type(attack_result) == FailedAttackResult:
            print("Attack failed.")
            original_text = attack_result.original_text.text;
            original_label = label_map[attack_result.original_output];
            original_score = 0
            result = {
                "success": False,
                "original_text": original_text,
                "original_label": original_label,
                "original_score": original_score
            }

            return flask.jsonify(result)

        elif type(attack_result) == AttackResult:
            print("Attack successful.")
            original_text, perturbed_text = attack_result.diff_color("html");
            original_label = label_map[attack_result.original_output];
            original_score = 0

            perturbed_label = label_map[attack_result.perturbed_output];
            perturbed_score = 0

            num_queries = attack_result.num_queries;

            result = {
                "success": True,
                "original_text": original_text,
                "original_label": original_label,
                "original_score": original_score,
                "perturbed_text": perturbed_text,
                "perturbed_label": perturbed_label,
                "perturbed_score": perturbed_score,
                "num_queries": num_queries
            }

            return flask.jsonify(result)

        else:
            return internal_error(500)

    return app

# Needed for AWS EB
application = create_app()

# run the app.
if __name__ == "__main__":
    application.run(debug=DEBUG, host="0.0.0.0", port=8000)

