from flask import Flask, request, render_template, redirect, url_for
from main import AI_Speech_Chatbot as chatbot

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        message = request.form.get('user_message')
        # Pass message as a query parameter to second page
        return redirect(url_for('second', message=message))
    return render_template('index.html')

@app.route('/second', methods=['GET', 'POST'])
def second():
    if request.method == 'POST':
        user_input = request.form.get('user_message')
        response = f"You asked: {user_input}"
        #(AI response would go here)
        ai_chatbot = chatbot()
        # print(ai_chatbot.greet())
        ai_reply = ai_chatbot.respond(user_input)
        #print(" AI Reply: ", ai_reply1)
        ai_reply = 'Response: ' + ai_reply
        return render_template('second.html', message=request.args.get('message', ''), response1=response, ai_reply=ai_reply)
    
    message = request.args.get('message', 'No message provided.')
    response = None
    return render_template('second.html', message=message, response1=response, ai_reply=None)

if __name__ == '__main__':
    app.run(debug=True)
