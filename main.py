from llm_code_1 import RAGSystem

class AI_Speech_Chatbot:
    def __init__(self):
        self.name = 'AI_Speech_Chatbot'
        self.rag_system = RAGSystem(
            directory_path="data/book/",
            index_name="codebase-rag",
            namespace="aurvedic_medicine_pdf"
        )
    def greet(self):
        return "Hello! I'm your AI Speech Chatbot. How can I assist you today?"
    def farewell(self):
        return "Goodbye! Have a great day!"
    def get_name(self):
        return self.name
    def respond(self, user_input):
        # Here you would implement the logic to generate a response based on user input
        # For simplicity, let's assume we have a function `chatbot_response` that does this
        return self.chatbot_response(user_input)
    
    def chatbot_response(self, user_input):
        # Very basic logic
        if "hello" in user_input.lower():
            return self.greet()
        elif "bye" in user_input.lower():
            return self.farewell()
        else:
            return self.rag_system.run(user_input)

if __name__ == '__main__':
    chatbot = AI_Speech_Chatbot()
    print(chatbot.greet())
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print(chatbot.farewell())
            break
        response = chatbot.respond(user_input)
        print(f"{chatbot.get_name()}: {response}")
      