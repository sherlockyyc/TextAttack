Web Visualization App for TextAttack
==========================

The goal TextAttacl-web-viz is to perform a similar function to Google's TensorFlow Playground, but for visualizations of evasion attacks in adversarial machine learning on text data.  This is a web service that enables the user to visualize the creation of adversarial samples to neural networks. Some notable features include:

1. Position highlighter
2. Importance heatmap
3. Togglability + interactiveness
4. Modular structure


How to Add Code
----------------

The code structure for TextAttack Playground is very simple, modular, and easy to add to.

<strong>Backend code </strong>(machine learning algorithm): place all relevant files in the webapp/models folder. Connect the model to the frontend code in the views.py file. Follow the example given for the deepwordbug model. 

<strong>Frontend code </strong>(for the actual visualization): place this in the webapp/templates folder. 

The about.html file provides a guideline for how to achieve all the features listed above. The index.html file demonstrates textAttack via webapp demo. There are extensive comments throughout each file to show exactly how and where code should be added. 



Installation
------------

If you choose to install this locally and not use the AWS EB link above, there are git submodules in this repository; to clone all the needed files, please clone this entire repository, then cd into the web_viz folder. 

The primary requirements for this package are Python 3 with textAttack library.  The `requirements.txt` file contains a listing of the required Python packages; to install all requirements, run the following:

```
pip3 -r install requirements.txt
```

If the above command does not work, use the following:

```
pip3 install -r requirements.txt
```

Or use the following instead if need to sudo:
```
sudo -H pip  install -r requirements.txt
```

Use:
----

### To Deploy the webserver:

Once you've downloaded the repo, cd into textAttack-web-viz, then run `python3 run.py` :

```
$ cd web-viz
$ python3 application.py &       
```

Or run the following command to run the webapp in the background even when logged out from the server:
```
$ cd web-viz
$ nohup python3 application.py &        # run in background even when logged out
```

Now use your favorite explorer to navigate to `localhost:9000`  or 'your_server_url:9000'
