import random
import os
import time
from pyChatGPT import ChatGPT

CHATGPT_MESSAGE = "a 5 paragraph text on *, narrated by david attenborough. include the scientific name. Make funny nature jokes."
PROJECT_PATH = "attenborough2_texts"
KEYWORDS_FILE = "keywords2.txt"
HARVEST_RANDOMLY = False
KEYWORD_STARTING_INDEX = 779

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

        print("USING KEYWORD: {} {}/{}".format(keyword, keyword_index, keyword_length))
        
        session_token = 'eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..pV3L-DtEQMUucq-K.wpRrgFtIOeRTCEdKZmNrGhQ9H9p7BBUku1O2c26ajzd3SPJWAICxVvPZLdfW4gjxktGNCmwuRnV-uGuBPjbnoixWkWfh0qr7BJmNxKGjVIrUQfSxiHTh7PJksu9wVrvQSVVDzUI9rGhfU8csupYh7guisX4oxIih6d5pfAX6V0ssCcFxCBoH5aPxihzN3sc1ed01YPwithsyxkZgKASalM3a2IBLNa3_l_wR0hNrqlwPk_aUL6gyvGwrlHfflySBxPiY_Fwx9Sohr93mvJ9iW6eZ2GhZMNJYrB8c_E3hhzlykWKL2aFneG_1JgMVUxJP_NzX1RnDO3pjXuEvNky9itpUbBSyS5BTl-FcY5KE_R5-SodLWfjwGa25O77wckeu_XfmqKuXd6GPp1GUO0Hlh_oJNxGtVwLs2446vccAjw7iZSRb6rOJo0cveYYu8H0i9ypCFJgh-rLTqN6zb3liFOTdWrJ39i99OZuhOKnLCUQzw9soCCKB70zTHDGxqnpzVsVWL2qhYlX9Ekaw-8NNRBTotiFYLlR4BM2O8_ZK9jeYZMDjP9gAnFn0Ld3Du0L7usYImHV0bd19PmASo6PDnoy0Mar2HgznT5S6C_fzzgegtnE4SLmWv6hvkfhUK7XJcuDHfszfvIntvMNRlplmNH3pGNpw_TNk6Px6BfcKFVZW4SIrgQxAo6-W2NTK8JFAIu3i5OpQrM3g5OkcIGRtKPU8hGWBlUwLFhHjUM7NGc2ia92xiDMgs0Zjjdxg4_4_WY8LEVsX0sct4jDlgeENSgKliZRF9jN4W170jKSsB9Dsowrt7V2H-OhEsFvc3PTUJ1u9w8SLwfXkFBjwRl2qn1FmomRuDuOwBaayHgSNEunS7h72OISlvz5mYk9RFxG1yksxG-TTbJePlht3v0BAx7BYpX0oRVJTlxereuGgyWimqLUnXTKDP7sIL51xqr_4qWXyNxtqA_dkBS-DrVfTwj8cLI2EOsP198GcOEzUpmhW9cxF1P8mXGsLykwNnDkkGwbqC1mRn9Gx1qYqKsqrD3AolX_g7oLeF64jjj6fNMwuHAEY1mSehjgdWz4HQLf5PXXqlqKv-A80yzo8RgliAKMtbLGHmnnFtzOOKe2bU-bI7f3u_YBs3-7Z4vyEuLiIaE0fMjMBvveji1MUvx20FIVnXgNIqFcHmg8v3jcyPMfeT7GKc_bCOHzf2qlHswQ8ud6dLGIQ5P7pYwsEppq6oy-mOkWjxdKdJjyn3eC-v6DDdnsQ5isylZ5M9PANjQEFISxETU3GEAGWjdJfS6H2V10LBvtoN7dv3Rf-dqvpWacqURQiabXTZ01ZP2zha5YjMdc9EXd4Zmznc7mQXRCO62lgVlIHxClGA9PyBzZl2oTgw5q7dwhGCCDf1oIrFEHSdYfP_sxGIUcnYSnzH3I4nBrlQwQqGv4h9LZnuUbHkMWDs7VF5wLYZAD0aTR0XAyCScvzFJB2pitkQxU12zntIhm6XJ2NjRTVWOFPUSyB3s9oR06JmfEoknWrSlAgW5OEzC9NKQDFVj3iuZ998P2z_HMS4wtkZDaX0JkgexxjzbNXyp-kRzGWYBAUuD1HuHadpbXpgO4FgwqMnx7mjWCEjbi-AAlPcLomzd2PITjxEs6REtFrM5joS4kuYhra9WA_uySEwQkbxbfh2yQ63kYHRrmeSwsIYDmWRAC5F7BuuPPJneueSuPVZPqDcg-b4yZ8Z_BlcFMNbIXvBW_aiuxbDWBOsVpWEW2m4ieviMdGLzTCoZPa6XH0gHTcNuynscovym9tWrwXm_Gv55H-GeoY73jalanq17yPVzmmNBPMkHTNonkmHruRPBM-C9QR4RiJgxYSGEI6r3yyMS7tfAbOTwX3Z6jbhBXvidetsRZKSyp3ZPrf8_ECrlfhOU3BhdJTxoHz2Xdq4QLNoQcE9e7oWMktSW6qzjG8cPJfy3sxwlF_1EwA_jKzPtIJOHZD6t4Pzzv7ckzWFagNB7ExWpf1G73-weF0qyy63HjY1BNGb4vccpN2tisnPlb0ZHsxo505t-2aa9k_sELIQyMd5Bc_0RwzIi0Ep8k5w7IPmixG0Y44j-P0ONq110ifZbpt_ZHuNfbI7YMwi8pFsLFzlFT4kt88sUA4DtmUto7d9i1R_nUMX4SOpHVAIBZ8rhdRndwT4ndEdWwe1ac_3ayYsWSfXzYWMOuHtxi7uclIGBe06L49m4IqLYB_PkVnWHqF3usIpuryTAsJ1hLSjoYnO4sb5nLugz5T6eMMTXhwutzgOw3NmFCtXa74Z6feFZHTD6yoLHyE4SapSDXY0mQfa__KLs8IKH-1M_JxFlniFIn0_LDJKo6XKRy8rHEjcKktmkmju-F2ukp_-6qS3DVeT6SKlQf7vH0iZqTBebg4spYCTv49XOPrj4eDCDBhaoQzQ3GSJk_j9n4Sdh2StscXNs1_oacKL2RH7K1__YMvdoJVxno1WcSxg9VgjJZs-BpUg1pkPNEall7zHV4z1ookzepfwZS_v6so0sYkJ6RQJerSGfBiAHErs1HYf_UcsX0A7m4WcVWDwDuJenghJrfa28zh9cal1sYbwr3oxnPFzEr8tVNGq50piCY6ThaLPqEeWNWx5ToM21Lf7hQ08APOvpIMBVfE_IftuDffXHZLq5VvrKQstWyy25BQKBvNFec.uNwBFttBNqTC1aCWDT9CkA' 
            
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
        
        message = message.replace('́', '') #weird chatgpt chars
        filename = "{}/{}.txt".format(subdir, i)
        with open(filename, 'w') as f:
            f.write("{}\n".format(keyword))
            f.write(message)

        api.__del__()
        time.sleep(10)

harvest_chatgpt()