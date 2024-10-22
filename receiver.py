import zmq

class Receiver:
    def __init__(self, url="tcp:127.0.0.1:7777"):
        self._url = url

    def run(self):
        ctx = zmq.Context.instance()
        subscriber = ctx.socket(zmq.SUB)
        subscription = "POSE"
        subscriber.setsocketopt(zmq.SUBSCRIBE, subscription)

        while True:
            topic, data = subscriber.recv_multipart()
            assert topic == subscription
            print(data)
