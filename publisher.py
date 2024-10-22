import threading
import time
from multiprocessing import Queue

import zmq as zmq


class Publisher:
    def __init__(self, host="127.0.0.1", port="7777"):
        self.host = host
        self.port = port
        # create a publishing server in a separate thread with a queue to communicate
        self.queue = Queue()
        server_thread = threading.Thread(target=self._run_server, args=())
        server_thread.start()
        pass

    def _run_server(self):
        # based on:  example for a pub-sub time server from tannenbaum - distributed systems,  page 202
        print("opening publishing-server at " + self.host + ":" + self.port)
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        communication_channel = "tcp://" + self.host + ":" + self.port
        socket.bind(communication_channel)
        while True:
            time.sleep(0.1) # wait for 0.1 seconds
            while not self.queue.empty():
                socket.send_string("POSE " + self.queue.get())  # publish pose als theme POSE

        pass

    def publish(self, pose):
        self.queue.put(pose) # put in queue to send it by the server in its own thread


