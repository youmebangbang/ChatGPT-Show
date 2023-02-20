import re
import random
import os
import time
from pyChatGPT import ChatGPT

CHATGPT_MESSAGE = "a 5 paragraph text on *, narrated by david attenborough. Make funny nature jokes. Include the scientific name."
PROJECT_PATH = "attenborough2_texts"
KEYWORDS_FILE = "keywords2.txt"
HARVEST_RANDOMLY = False
KEYWORD_STARTING_INDEX = 2442

try:
    keywords = []
    with open(KEYWORDS_FILE, 'r') as f:
        keywords = f.read().split(",") 
except:
    print("ERROR: NO KEYWORD FILE FOUND")
    exit()

def harvest_chatgpt():
    api = None
    keyword_index = KEYWORD_STARTING_INDEX
    keyword_length = len(keywords)
    while True:
        if HARVEST_RANDOMLY: 
            keyword_index = random.randint(0, keyword_length-1)
            keyword = keywords[keyword_index]
        else:            
            keyword = keywords[keyword_index]
            keyword_index += 1
            if keyword_index == keyword_length:
                keyword_index = 0
                print("HARVESTER REACHED END OF KEYWORD LIST")
                exit()

        print("USING KEYWORD: {} {}/{}".format(keyword, keyword_index, keyword_length))
        
        session_token = ''
            
        if api:
            api.__del__()
        api = ChatGPT(session_token)    
        text = CHATGPT_MESSAGE.replace("*", keyword)

        while True:
            try:
                print("SENDING CHATGPT MESSAGE")
                t = api.send_message(text)
            except:
                print("CHATGPT RUNNING OFFLINE")
                # chatgpt timed from too many requests                                    
                api.__del__()
                api = ChatGPT(session_token)  
                time.sleep(10)    
            else:
                break
            time.sleep(1)
         
        message = t['message']
        
        # write message and keyword to file
        os.makedirs(PROJECT_PATH, exist_ok=True)
        subdir = f"{PROJECT_PATH}"
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        i = 0
        while True:
            filename = "{}/{}.txt".format(subdir, i)
            if not os.path.exists(filename):
                break
            i += 1
        
        # message = message.replace('́', '') #weird chatgpt chars
        message = message.replace('\n', ' ')
        pattern = r'[\w\s\.,\-\'!?]+'
        message = ''.join(re.findall(pattern, message))

        filename = "{}/{}.txt".format(subdir, i)
        with open(filename, 'w') as f:
            f.write("{}\n".format(keyword))
            f.write(message)

        api.__del__()
        time.sleep(10)

harvest_chatgpt()
