import random
import os
import time
from pyChatGPT import ChatGPT

CHATGPT_MESSAGE = "a 5 paragraph text on *, narrated by david attenborough. use some funny nature jokes."
PROJECT_PATH = "attenborough2"
KEYWORDS_FILE = "keywords2.txt"


try:
    keywords = []
    with open(KEYWORDS_FILE, 'r') as f:
        keywords = f.read().split(",") 
except:
    print("ERROR: NO KEYWORD FILE FOUND")
    exit()

def harvest_chatgpt():
    api = None
    while True:
        i = len(keywords)
        keyword = keywords[random.randint(0, i-1)]
        print("USING KEYWORD: {}".format(keyword))
        
        session_token = 'eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..yrDB2Ni-wsZs_XO6.djJ2xe7jUrxTOYXpu5gogTBmmcw2b_lC-1Msxu4Chh4XdcLPQHtY66AndVhthqjVNaL-kzIu6DugUeIuO_g0wK5acYGWdbm24yPl18H-szA7fWicl6-ZVEiNfl3JcuLGYR7boTdQ8tPp6z_jqacPDns8w0lDNEAhWk09C5OUMKmEyohKHgP3rgEBpa_tKjutJzLIcQUe-zhgCg7iBoA38yUGlVZuSlwiP7FMGdvk6dFMNZczZ4iOOjDdsRxmaFxta23MKBBL-A7M5Af3Ocesf4TgIb2PH8nQfYPfbsdcacAoWKxcIuIh2vHzxigb8Ma-lb1Vh2_lTWFKCfJ-0vQKsu5cqhaIUap6f1E-gt_nYAsQ6XF847nq-lrinKqvHQEZPQLy_seozpYba0FpPOYj44wfmGxQSqGJd7S75tjNVLPnJolqP9i-NXrWeuTcUgiO_Q7pNlZS8hV0nx_c1n15hCoUAN3dqsjZ--_aRnT4e1UkWt1W1BoYG46dgiBcuLhcccuYEXLqJqIex6DVGn2gt0buMrNhyzzsnsGcbU5AgXCqoPxr8ZhRZgiI1-3UfonPGEta-rOWO16e3N4MtfVKexhlcvncE2DwhqzDTnQCRJr7AZ5qGB6rJz6Akov0ywyQ0RyL1rmGbUrh6H2WH1T0suCZpahf6RIOcQY35rb1QdoH4CTjqud-seedByKcr6znQMjL-fe6RUtiQzaBwOYMYxuOYy5G0Z8R_hA6gVqVsy3Z5x9Yw1b0uIGSpE3XSkTUCjdzmng8SE9ydzCkcczaLFbV57M5b4vQbPwZR8EhMd6qDX-v_KoDlbWGIeFfp3KfGKYVbokMHRbm0bJgeggFJPwhjOb-lvKjojvPJDEBgd4cMlWml2I1KLLYhwN8C3DPJZqeTxfZwAgVpjxQQ5ScYqPiG7kTcM-te8vmUDsTMWX5vXjDlJKyhyIOMu5OfO3mqo3-No40zG8aNMQulUTX_XCVE8vSbIti5Qs87VjhS70ZAaSXo8QK5TjKT9VerID7Bl4u1I1CMXMiYLdk4fZhydSG4lKvHamOYSAKg-RL0DpbAG6Y3c-ftPFL7DKKqpjAAUm4HS1_eW52CoE0q3_ON_Yk_gTrXwLSS8oOfaZ_48rAPoHEcLODezD-xA4SdffBFtIynM9YGR5QBmiKWyPTX_X_K_9MLkYS4vzP-UHp6HIJctoLoztD_71wVhP6eOpybKUtPSD-8dFLdef3wXjLCGJKO3mx0G07icrit56StANR0VzFLRlaBkwxekk-74ec1X8vVzCTARjh-t5P02HD7kR9Z_Od9LKZ5dUnG1pgVXKSV87cdInLuU1pVcVlbEENec-Mc6ES2UQu3SWG8qBkziFHUl0nyUb4_8R4FBRy-sepba-zoN_UveLrLbLUn1LDCuKMRsYYwOUUPYWffcHil4fuan1j9hGZZsUJ9Dt0LmiByp9Ng-zW1UFOzt8uy-cUnl9FkdStC2-AvGKZ-PquRnuDpZcqsJ0pkEWwsBi9byoKYBSAcy259cHPA6mgxGljuv8K0IhQtJwdf-mOcrHgKJmikSM4yN9uB5eOcyLQB7a98GckUPwQRSWojyI6-lVTaKQIDH7em-PerKqj7X-WOeNjH5FTlk7ChuI7TzifXFOtyefbkf_wGMfu9CN_A1IaWbuzuYE71m4Tn_V0QyYJz8Pkhj9HAFrv_zTRcYNnTP8gUW745txwFUPTW_0wqQkuMmtJ3VoM2xmgJ81F3QavEy4VGC22nGz2RwuN0YQwDmRLo3rpWkEk5L6vxxe-Ek4B_BW9B79_MBbe2lXG0Ac5x9qAadZb7z-AKpfF_4VFi5g4ge9poYGJSlKTbkOaNgdDgTz2dhA786tv5ZmUiC6Yom4J0aKrVRQgxtAZvQTbQ4rERku02ATTW3fbu-srzbRzDxVWfUFUsrpePFTDpK0XJIdBrEeTeMAQiD4jwiG1TqP-jDKYlmUByNjuxXeKaHqzm6IKIXmDKr4ld3_UaE_cbZJfszsW7u8ZyhZ-CJwnCPUDcLVnntj2BR3r0JGOweNAuIISBJks5r6jM4pWDrBhPZ4G9T4Q5eK4vImUvmYVhgoaCuEF7Ux6LuE3U8f5Ym2EQL8FfCyCNPhlY1BjxSRicdpudnqU2cxdmxqW7Tb41IZonVYy6E9oh93X6SwEH37kASoXvxOb3rJ7b7T6ZsjQTiScA7HJeBZfs174TP_Bq7tnFCm35t0lHmL61Wwtc-ob1m9JpUruiI0lZ1jaksCWEitbVgbpS6o6eo7w-xjr1hzosQ9_pUpxmjpzpfbixILHnH_y6dxXcruJCYWIEXkAYeUpfgVcnYX9oOTApQLyyVO6KhY3i3ovmzsyhbuBrBUsC8LJDs0nTmcgINi5M9Ss53Z4kEIxIHlJvErmYE-NjOrD6Sc1yHbKWsSqtmXpESdEXKipGY5thAvvNOf-Zta7TGmXdUwf_RTnrcDw3Zojrhzlz8UyuA2JnjDocoDs8y89G_prTWtZ6Yj2pvDhxyoTEZXrpihEydwYIILXSjXSMpOAFXM9dNa_4kq0bmTpXEm2VkNu_IqXljnau7k-CVvIHQoDo2sViXewmh6vPkuXlKZYVgEmswvf3Qd-hcQNlPxzcuIhSefXz2VZV_fvu0Y6f2BuyfnVHUOs219MBr-ruVkPaxfbbQEtlV0yE_o.vTYR3P2e8MVXamfAOvAjew' 
            
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
            else:
                break
            time.sleep(60)

        message = t['message']
        
        # write message and keyword to file
        os.makedirs(PROJECT_PATH, exist_ok=True)
        subdir = f"{PROJECT_PATH}_texts"
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        i = 0
        while True:
            filename = "{}/{}.txt".format(subdir, i)
            if not os.path.exists(filename):
                break
            i += 1
        
        filename = "{}/{}.txt".format(subdir, i)
        with open(filename, 'w') as f:
            f.write("{}\n".format(keyword))
            f.write(message)

        api.__del__()
        time.sleep(30)

harvest_chatgpt()