# End-to-end-ML-Airbnb-Price-Predictor

"This application is being developed as part of an end-to-end machine learning project, inspired by my Data Science Master's Degree course at University of Miami"

In the university we were just required to Clean, perform EDA, preprocess using different techniques such as splitting, apply subset selection techniques (lasso), run the different models that we had learnt.

After few years of further study, I have been wondering what I had done, and had questions of what is the actually purpose, so I decided to study Data Engineeing, basics O sfotware engineering and basics of devops, and reached a conclusion that it is a cycle, from that I started working on transcending the notebooks which is where Data Scientist excel and started to think more and more about transitioning to Machine Learning Engineering.

With all of these knowledge I grabbed on my for DS project and tried to apply the techniques from DS and ML that I had learnt, with the goal of making my models ready for deployment into prodcution, marking the first project of many in hopes to follow a life as AI|ML Engineer.

# Work structure

END-TO-END-AIRBNB-PRICE-PREDICTOR
├───.github
│ └───workflows
├───catboost_info
│ ├───learn
│ └───tmp
├───configs
├───data
│ ├───processed
│ └───raw
├───deployment
│ ├───Kubernetes
│ └───MLFlow
├───models
│ └───trained
├───notebooks
│ └───catboost_info
│ ├───learn
│ └───tmp
└───src
├───api
├───data
├───features
└───models

# Running this specific app commands

1- docker image build
2- docker ps -l
3- docker image ls
4- docker compose up -d
5- docker compose ps
6- docker compose logs
7- localhost:8000/docs

Made by MelvinMaria
