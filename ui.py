import gradio as gr

import Graph
# Initialize counter for message numbering
counter = 1

# Function to process input and return the conversation history
def chatbot_response(user_input, history):
    global counter

    # If history is None (first interaction), initialize it as an empty list
    if history is None:
        history = []

    # Generate bot response
    x,y=Graph.run_loop(user_input)
    print(type(x))
    #bot_response = "THE SIMILARITY SEARCH RESULTS : "+y+"\n\n\n"+x
    bot_response = x
    # Increment counter for the next message
    counter += 1

    # Append user and bot responses to the history
    history.append(("User", user_input))
    history.append(("Bot", bot_response))

    # Return updated history
    return history, history

# Gradio interface setup
iface = gr.Interface(
    fn=chatbot_response, 
    inputs=[
        gr.Textbox(label="Enter your message", placeholder="Ask me anything...", lines=1),  # User input
        gr.State()  # Keeps track of chat history
    ],
    outputs=[
        gr.Chatbot(label="Chat History"),  # Display chat history as bubbles
        gr.State()  # Updated chat history for further interaction
    ],
    live=False  # Ensure that the function is triggered only after user submits input
)

# Launch the interface
iface.launch()
