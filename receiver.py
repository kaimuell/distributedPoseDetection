import zmq
from converter import Converter

class Receiver:
    def __init__(self, url="tcp://127.0.0.1:7777"):
        self._url = url
        self._converter = Converter()

    def run(self):
        ctx = zmq.Context()
        subscriber = ctx.socket(zmq.SUB)
        print("connect to socket")
        subscriber.connect(self._url)
        subscription = "POSE"
        print("set socket opt")
        subscriber.setsockopt_string(zmq.SUBSCRIBE, subscription)

        while True:
            recieved = subscriber.recv_string()
            topic, data = recieved.split()
            assert topic == subscription
            data = self._converter.dataframe_from_json(data)
            print(data)
