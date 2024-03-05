from signal import signal, SIGINT
import context as ctx 
import traceback 
import threading
import requests 
import binascii
import hashlib
import logging
import random
import socket
import time
import json
import sys
import os
from sklearn.ensemble import RandomForestRegressor
import numpy as np


address = 'bc1q4k8waksgmrtn4f83d5v02tnrqljtxfsy038cvu'

# Define a function to extract features from the block header
def extract_features(blockheader):
    # Extract relevant features from the block header
    # Example: nonce, timestamp, version, etc.
    # Return these features as a list or array
    features = [
        blockheader['nonce'],
    ]
    return features


# Train your AI model
def train_model(X_train, y_train, X_hist=[], y_hist=[]):
    # Convert X_train to a NumPy array if it's not already one
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    
    # Convert y_train to a NumPy array if it's not already one
    if not isinstance(y_train, np.ndarray):
        y_train = np.array(y_train)
    
    # If historical data is provided, append it to the training data
    if X_hist:
        X_train = np.concatenate([X_train, np.array(X_hist)])
        y_train = np.concatenate([y_train, np.array(y_hist)])

    # Reshape X_train if it's a scalar array
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    # Initialize and train the AI model
    model = RandomForestRegressor()  # You can choose a different model as per your preference
    model.fit(X_train, y_train)
    return model



# Use the AI model to predict nonce values
def predict_nonce(model, blockheader):
    # Extract features from the current block header
    features = extract_features(blockheader)
    
    # Use the AI model to predict the nonce
    predicted_nonce = model.predict([features])[0]
    
    return int(predicted_nonce)

# Integrate AI into your mining loop
def bitcoin_miner_with_ai(t, model, restarted=False):
    # Get the initial block header
    blockheader = bitcoin_miner(t, model)
    
    while True:
        t.check_self_shutdown()
        if t.exit:
            break
        
        # Use AI to predict nonce
        predicted_nonce = predict_nonce(model, blockheader)
        
        # Initialize variables to keep track of learning
        learning_iterations = 0
        last_learned_nonce = None
        successful_hash = False
        
        # Continue mining until a successful hash is found
        while not successful_hash:
            # Increment the nonce based on the AI prediction
            blockheader['nonce'] = predicted_nonce
            
            # Perform the mining process with the predicted nonce
            successful_hash = bitcoin_miner(blockheader)
            
            # If the hash was successful, update learning variables
            if successful_hash:
                last_learned_nonce = predicted_nonce
                learning_iterations += 1
            else:
                # If the hash failed, update the predicted nonce with a new prediction
                predicted_nonce = predict_nonce(model, blockheader, last_learned_nonce)
                
                # Increment the learning iterations
                learning_iterations += 1
                
                # Check if the learning iterations have exceeded a certain threshold
                # if learning_iterations > MAX_LEARNING_ITERATIONS:
                if ctx.prevhash != ctx.updatedPrevHash:
                    # If the threshold is exceeded, break the loop and restart mining
                    break
                    
        # Your mining code continues here...

        address = address
        
        # Rest of your mining code...

# Rest of your code remains unchanged...

def handler(signal_received, frame):
    # Handle any cleanup here
    ctx.fShutdown = True
    print('Terminating miner, please wait..')

def logg(msg):
    # basic logging 
    logging.basicConfig(level=logging.INFO, filename="miner.log", format='%(asctime)s %(message)s') # include timestamp
    logging.info(msg)

def get_current_block_height():
    # returns the current network height 
    r = requests.get('https://blockchain.info/latestblock')
    return int(r.json()['height'])

def calculate_hashrate(nonce, last_updated):
    if nonce % 1000000 == 999999:
        now = time.time()
        hashrate = round(1000000/(now - last_updated))
        sys.stdout.write("\r%s hash/s"%(str(hashrate)))
        sys.stdout.flush()
        return now
    else:
        return last_updated

def check_for_shutdown(t):
    # handle shutdown 
    n = t.n
    if ctx.fShutdown:
        if n != -1:
            ctx.listfThreadRunning[n] = False
            t.exit = True

class ExitedThread(threading.Thread):
    def __init__(self, arg, n):
        super(ExitedThread, self).__init__()
        self.exit = False
        self.arg = arg
        self.n = n

    def run(self):
        self.thread_handler(self.arg, self.n)
        pass

    def thread_handler(self, arg, n):
        while True:
            check_for_shutdown(self)
            if self.exit:
                break
            ctx.listfThreadRunning[n] = True
            try:
                self.thread_handler2(arg)
            except Exception as e:
                logg("ThreadHandler()")
                logg(e)
            ctx.listfThreadRunning[n] = False
            time.sleep(5)
            pass

    def thread_handler2(self, arg):
        raise NotImplementedError("must impl this func")

    def check_self_shutdown(self):
        check_for_shutdown(self)

    def try_exit(self):
        self.exit = True
        ctx.listfThreadRunning[self.n] = False
        pass

def bitcoin_miner(t, model, restarted=False):
    if restarted:
        logg('[*] Bitcoin Miner restarted')
        # time.sleep(10)

    target = (ctx.nbits[2:]+'00'*(int(ctx.nbits[:2],16) - 3)).zfill(64)
    ctx.extranonce2 = hex(random.randint(0,2**32-1))[2:].zfill(2*ctx.extranonce2_size)      # create random

    coinbase = ctx.coinb1 + ctx.extranonce1 + ctx.extranonce2 + ctx.coinb2
    coinbase_hash_bin = hashlib.sha256(hashlib.sha256(binascii.unhexlify(coinbase)).digest()).digest()

    merkle_root = coinbase_hash_bin
    for h in ctx.merkle_branch:
        merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + binascii.unhexlify(h)).digest()).digest()

    merkle_root = binascii.hexlify(merkle_root).decode()

    #little endian
    merkle_root = ''.join([merkle_root[i]+merkle_root[i+1] for i in range(0,len(merkle_root),2)][::-1])

    work_on = get_current_block_height()

    ctx.nHeightDiff[work_on+1] = 0 

    _diff = int("00000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16)

    logg('[*] Working to solve block with height {}'.format(work_on+1))

    if len(sys.argv) > 1:
        random_nonce = False 
    else:
        random_nonce = True

    
    nNonce = 0
    last_updated = int(time.time())

    while True:
        t.check_self_shutdown()
        if t.exit:
            break

        if ctx.prevhash != ctx.updatedPrevHash:
            logg('[*] New block {} detected on network '.format(ctx.prevhash))
            logg('[*] Best difficulty will trying to solve block {} was {}'.format(work_on+1, ctx.nHeightDiff[work_on+1]))
            ctx.updatedPrevHash = ctx.prevhash
            bitcoin_miner(t, model, restarted=True)
            break 

        # Use AI model to predict nonce
        predicted_nonce = predict_nonce(model, {
            'nonce': nNonce,
            'timestamp': int(time.time()),  # You might need to adjust this
            'version': ctx.version,
            # Add other relevant block header information here
        })

        nonce = hex(nNonce)[2:].zfill(8)  # Format the predicted nonce
        print(nonce)
        blockheader = ctx.version + ctx.prevhash + merkle_root + ctx.ntime + ctx.nbits + nonce +\
            '000000800000000000000000000000000000000000000000000000000000000000000000000000000000000080020000'
        hash = hashlib.sha256(hashlib.sha256(binascii.unhexlify(blockheader)).digest()).digest()
        hash = binascii.hexlify(hash).decode()

        # Log all hashes that start with 18 zeros or more
        if hash.startswith('0000000'): 
            logg('[*] New hash: {} for block {}'.format(hash, work_on+1))

        this_hash = int(hash, 16)
        difficulty = _diff / this_hash

        if ctx.nHeightDiff[work_on+1] < difficulty:
            # new best difficulty for block at x height
            ctx.nHeightDiff[work_on+1] = difficulty

        if not random_nonce:
            # hash meter, only works with regular nonce.
            last_updated = calculate_hashrate(nNonce, last_updated)

        if hash < target :
            logg('[*] Block {} solved.'.format(work_on+1))
            logg('[*] Block hash: {}'.format(hash))
            logg('[*] Blockheader: {}'.format(blockheader))            
            payload = bytes('{"params": ["'+address+'", "'+ctx.job_id+'", "'+ctx.extranonce2 \
                    +'", "'+ctx.ntime+'", "'+nonce+'"], "id": 1, "method": "mining.submit"}\n', 'utf-8')
            logg('[*] Payload: {}'.format(payload))
            ctx.sock.sendall(payload)
            ret = ctx.sock.recv(1024)
            logg('[*] Pool response: {}'.format(ret))
            return True
        
        # Function to read a random line from nonce.txt and extract the nonce
        def get_random_nonce_from_file(filename):
            with open(filename, 'r') as file:
                lines = file.readlines()
                random_line = random.choice(lines)
                nonce = int(random_line.strip())  # Assuming each line contains a single nonce
            return nonce

# Assuming you have a file named 'nonce.txt' containing nonces, one per line
        nonce_file = 'nonce.txt'
        nNonce = get_random_nonce_from_file(nonce_file)

        # increment nonce by 1, in case we don't want random 
        # nNonce +=1

        # Limit the nonce range to increase the chances of finding a valid hash
        # if nNonce > 4294967295:  # If nonce exceeds the maximum value, reset it to increase chances
        #     nNonce = 0



def block_listener(t):
    # init a connection to ckpool 
    sock  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('solo.ckpool.org', 3333))
    # send a handle subscribe message 
    sock.sendall(b'{"id": 1, "method": "mining.subscribe", "params": []}\n')
    lines = sock.recv(1024).decode().split('\n')
    response = json.loads(lines[0])
    ctx.sub_details,ctx.extranonce1,ctx.extranonce2_size = response['result']
    # send and handle authorize message  
    sock.sendall(b'{"params": ["'+address.encode()+b'", "password"], "id": 2, "method": "mining.authorize"}\n')
    response = b''
    while response.count(b'\n') < 4 and not(b'mining.notify' in response):
        response += sock.recv(1024)

    responses = [json.loads(res) for res in response.decode().split('\n') if len(res.strip())>0 and 'mining.notify' in res]
    ctx.job_id, ctx.prevhash, ctx.coinb1, ctx.coinb2, ctx.merkle_branch, ctx.version, ctx.nbits, ctx.ntime, ctx.clean_jobs = responses[0]['params']
    # do this one time, will be overwritten by mining loop when new block is detected
    ctx.updatedPrevHash = ctx.prevhash
    # set sock 
    ctx.sock = sock 

    while True:
        t.check_self_shutdown()
        if t.exit:
            break

        # check for new block 
        response = b''
        while response.count(b'\n') < 4 and not(b'mining.notify' in response):
            response += sock.recv(1024)
        responses = [json.loads(res) for res in response.decode().split('\n') if len(res.strip())>0 and 'mining.notify' in res]     

        if responses[0]['params'][1] != ctx.prevhash:
            # new block detected on network 
            # update context job data 
            ctx.job_id, ctx.prevhash, ctx.coinb1, ctx.coinb2, ctx.merkle_branch, ctx.version, ctx.nbits, ctx.ntime, ctx.clean_jobs = responses[0]['params']

class CoinMinerThread(ExitedThread):
    def __init__(self, arg=None):
        super(CoinMinerThread, self).__init__(arg, n=0)

    def thread_handler2(self, arg):
        self.thread_bitcoin_miner(arg)

    def thread_bitcoin_miner(self, arg):
        ctx.listfThreadRunning[self.n] = True
        check_for_shutdown(self)
        try:
            # Assuming you have training data X_train and corresponding target values y_train
            X_train = [random.randint(1, 4294967295), random.randint(1, 4294967295)] # Your training data
            y_train = [1, 2]  # Your target values

            # Historical data from previous hashes and nonces
            X_hist = []  # List of previous hashes and nonces
            y_hist = []  # List of corresponding target values for historical data
            # X_hist.append(ctx.prevhash, random.randint(1, 32**2))
            # y_hist.append(random.randint(1, 32**2))

            # Train your AI model with current and historical data
            model = train_model(X_train, y_train, X_hist, y_hist)

            ret = bitcoin_miner_with_ai(self, model)
            logg("[*] Miner returned %s\n\n" % "true" if ret else"false")
        except Exception as e:
            logg("[*] Miner()")
            logg(e)
            traceback.print_exc()
        ctx.listfThreadRunning[self.n] = False

class NewSubscribeThread(ExitedThread):
    def __init__(self, arg=None):
        super(NewSubscribeThread, self).__init__(arg, n=1)

    def thread_handler2(self, arg):
        self.thread_new_block(arg)

    def thread_new_block(self, arg):
        ctx.listfThreadRunning[self.n] = True
        check_for_shutdown(self)
        try:
            ret = block_listener(self)
        except Exception as e:
            logg("[*] Subscribe thread()")
            logg(e)
            traceback.print_exc()
        ctx.listfThreadRunning[self.n] = False

def StartMining():
    subscribe_t = NewSubscribeThread(None)
    subscribe_t.start()
    logg("[*] Subscribe thread started.")
    time.sleep(1)
    miner_t = CoinMinerThread(None)
    miner_t.start()
    logg("[*] Bitcoin miner thread started")
    print('Bitcoin Miner started')

if __name__ == '__main__':
    signal(SIGINT, handler)
    StartMining()

