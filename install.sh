#!/bin/bash
pip install -r requirements.txt
python setup.py

export JAVA_HOME=$HOME/.jdk/jdk-17.0.17+10
export PATH=$PATH:$JAVA_HOME/bin

python main.py
