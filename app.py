from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    message = None
    if request.method == 'POST':
        message = request.form.get('user_message')
    return render_template('index.html', message=message)

@app.route('/second')
def second():
    return render_template('second.html')

if __name__ == '__main__':
    app.run(debug=True)
