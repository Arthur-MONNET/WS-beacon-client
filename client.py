import websocket
import json

url = 'ws://localhost:8080'

def on_open(ws):
    print('WebSocket connected')
    ws.send(json.dumps({'type': 'balise'}))
    while True:
        user_input = input('message to send: ')
        ws.send(json.dumps({'type': 'balise', 'payload': user_input}))

def on_message(ws, message):
    print(f'Received message: {message}')

def on_close(ws):
    print('WebSocket disconnected')

ws = websocket.WebSocketApp(url, on_open=on_open, on_message=on_message, on_close=on_close)

ws.run_forever()
