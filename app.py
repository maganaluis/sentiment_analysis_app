from flask import Flask
from flask import request
from flask import render_template
import matplotlib.pyplot as plt
from joblib import dump, load
import numpy as np
import pandas as pd
import json
import io
import base64

model = load('sentiment.model')
vectorizer = load('vectorizer.model')
df = pd.read_csv('vocabp.csv')
df['positive'] = df['p-val'] > 0
df['p-val'] = df['p-val'] * -1
df = df.set_index('words')

def get_pred_plot(p):
    plt.bar(['Negative','Positive'], p.flatten())
    plt.ylabel('Probability')
    plt.title('Prediction')
    plt.tight_layout()
    output = io.BytesIO()
    plt.savefig(output)
    plt.clf()
    plt.cla()
    plt.close()
    return base64.b64encode(output.getvalue()).decode()

def get_vocab_plot(x):
    idx = list(np.argwhere(x.toarray().flatten() != 0).flatten())
    vwords = np.array(vectorizer.get_feature_names())[idx]
    tdf = df[df.index.isin(vwords)]
    if tdf.shape[0] == 0:
        return None
    tdf.head(50).plot(kind='barh', color=[tdf.positive.map({True: 'tab:red', False: 'tab:blue'})],
                                    figsize=(12,8))
    plt.xlabel('P-val')
    plt.title('Words in Vocabulary')
    plt.tight_layout()
    output = io.BytesIO()
    plt.savefig(output)
    plt.clf()
    plt.cla()
    plt.close()
    return base64.b64encode(output.getvalue()).decode()

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 #might want to be careful here

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    s = request.args['sentence']
    x = vectorizer.transform(np.array([s]))
    p = model.predict_proba(x)
    #  response = make_response(output.getvalue())
    #  response.mimetype = 'image/png'
    plot_url_pred = get_pred_plot(p)
    plot_url_vocab = get_vocab_plot(x)
    #  https://stackoverflow.com/questions/20836766/how-do-i-remove-broken-image-box
    return render_template('index.html', chart1=plot_url_pred, chart2=plot_url_vocab)

if __name__ == '__main__':
    app.run()
