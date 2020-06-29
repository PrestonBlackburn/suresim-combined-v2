from flask import Flask, render_template, redirect, url_for, request

##importing blueprint-
from shrinkage.shrinkage_csv import shrinkage_csv
from EOS.EOS_app import EOS_app
from scaling.scaling_app import scaling_app
from EOS_API.EOS_api_app import EOS_api_app


app = Flask(__name__)
app.register_blueprint(shrinkage_csv, url_prefix = "/shrinkage")
app.register_blueprint(EOS_app, url_prefix = "/EOS" )
app.register_blueprint(EOS_api_app, url_prefix="/EOS/API")
app.register_blueprint(scaling_app, url_prefix = "/scaling" )



@app.route("/")
def main():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug = True)