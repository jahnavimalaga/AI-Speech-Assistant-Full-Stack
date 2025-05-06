from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        message = request.form.get('user_message')
        # Pass message as a query parameter to second page
        return redirect(url_for('second', message=message))
    return render_template('index.html')

@app.route('/second')
def second():
    message = request.args.get('message', 'No message provided.')
    return render_template('second.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)
