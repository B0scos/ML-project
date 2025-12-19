from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


@app.route('/')
def index():
    return "Welcome to the Prediction API!"

