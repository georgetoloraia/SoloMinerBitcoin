# import random

fShutdown = False
listfThreadRunning = [False] * 2 
local_height = 0 
nHeightDiff = {}


updatedPrevHash = None



job_id = None 
prevhash = None 
# coinb1 = None 
# coinb2 = None 
merkle_branch = None 
version = None 
# nbits = None 
nbits = "1a44b9f8"
ntime = None 
clean_jobs = None 

sub_details = None 
# extranonce1 = None 
# extranonce2 = None
extranonce2_size = None

coinb1 = ""
coinb2 = ""
extranonce1 = ""
extranonce2 = ""



sock = None

# blockheader = random.randint(1, 4294967295)