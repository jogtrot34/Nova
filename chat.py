from fast_responder import FastResponder

print("Loading Nova chatbot...")
responder = FastResponder()
responder.load()
print("Ready! Type your question (or 'quit' to exit)\n")

while True:
    try:
        text = input("You: ")
        if text.lower() in ['quit', 'exit', 'bye']:
            print("Nova: Goodbye!")
            break
        
        reply = responder.respond(text)
        if reply:
            print(f"Nova: {reply}\n")
        else:
            print("Nova: I didn't quite catch that\n")
            
    except KeyboardInterrupt:
        print("\nNova: Goodbye!")
        break
    except EOFError:
        break
