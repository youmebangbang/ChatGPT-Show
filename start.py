from pyChatGPT import ChatGPT
from youtubesearchpython import VideosSearch
import shutil
import re
import os
import sys
import time
import random
import vlc
import torch
import numpy as np

sys.path.append('tacotron2/')
sys.path.append('waveglow/')

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
import soundfile as sf
import simpleaudio as sa
from pydub import AudioSegment 
from pydub.playback import play
import threading
import tkinter as tk

try:
    KEYWORDS = []
    with open("keywords.txt", 'r') as f:
        KEYWORDS = f.read().split(",") 
except:
    print("ERROR: NO KEYWORD FILE FOUND")
    exit()

PROJECT_PATH = "attenborough"
HARVESTER_ON = True
TACOTRON_MODEL_PATH = "attenborough_v4_checkpoint_4225700(5050500)"
WAVEGLOW_MODEL_PATH = "attenborough_v4_waveglow_2030200(3133000)1e-5"
CHATGPT_MESSAGE = "a 5 paragraph text on *, narrated by david attenborough. use some funny nature jokes."

class Control:
    def __init__(self):
        self.is_audio_playing = False
        self.is_scene_running = False
        self.is_inference_ready = False
        self.is_inference_running = False
        self.get_video_ready = False
        self.get_video_running = False
        self.is_play_audio_running = False
        self.wav_list = []
        self.vlc_obj = vlc.Instance("--no-xlib")
        self.vlcplayer = self.vlc_obj.media_player_new()
        
        self.next_episode_label = tk.Label(root, text="Generating next episode", font=("TkDefaultFont", 20), width=20, bg='black', fg='white')        
        self.title_label = tk.Label(root, text="", font=("TkDefaultFont", 15), bg='black', fg='white')
        self.video_title = ""
        self.video_link = ""
        
        self.keyword = ""
        self.chatgpt_message = ""        
        self.session_token = None        
        self.is_download_video_running = False

        self.taco_model = None
        self.waveglow_model = None
        self.search_text = ""

        hparams = create_hparams()
        hparams.sampling_rate = 22050
        hparams.p_attention_dropout = 0
        hparams.p_decoder_dropout = 0
        hparams.max_decoder_steps = 2000

        self.taco_model = load_model(hparams)
        self.taco_model.load_state_dict(torch.load(TACOTRON_MODEL_PATH)['state_dict'])
        _ = self.taco_model.cuda().eval().half()

        self.waveglow_model = torch.load(WAVEGLOW_MODEL_PATH)['model']
        self.waveglow_model.cuda().eval().half() 
    
    def harvest_chatgpt(self):
        while True:
            i = len(KEYWORDS)
            keyword = KEYWORDS[random.randint(0, i-1)]
            print("USING KEYWORD: {}".format(keyword))
            
            session_token = 'eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..yrDB2Ni-wsZs_XO6.djJ2xe7jUrxTOYXpu5gogTBmmcw2b_lC-1Msxu4Chh4XdcLPQHtY66AndVhthqjVNaL-kzIu6DugUeIuO_g0wK5acYGWdbm24yPl18H-szA7fWicl6-ZVEiNfl3JcuLGYR7boTdQ8tPp6z_jqacPDns8w0lDNEAhWk09C5OUMKmEyohKHgP3rgEBpa_tKjutJzLIcQUe-zhgCg7iBoA38yUGlVZuSlwiP7FMGdvk6dFMNZczZ4iOOjDdsRxmaFxta23MKBBL-A7M5Af3Ocesf4TgIb2PH8nQfYPfbsdcacAoWKxcIuIh2vHzxigb8Ma-lb1Vh2_lTWFKCfJ-0vQKsu5cqhaIUap6f1E-gt_nYAsQ6XF847nq-lrinKqvHQEZPQLy_seozpYba0FpPOYj44wfmGxQSqGJd7S75tjNVLPnJolqP9i-NXrWeuTcUgiO_Q7pNlZS8hV0nx_c1n15hCoUAN3dqsjZ--_aRnT4e1UkWt1W1BoYG46dgiBcuLhcccuYEXLqJqIex6DVGn2gt0buMrNhyzzsnsGcbU5AgXCqoPxr8ZhRZgiI1-3UfonPGEta-rOWO16e3N4MtfVKexhlcvncE2DwhqzDTnQCRJr7AZ5qGB6rJz6Akov0ywyQ0RyL1rmGbUrh6H2WH1T0suCZpahf6RIOcQY35rb1QdoH4CTjqud-seedByKcr6znQMjL-fe6RUtiQzaBwOYMYxuOYy5G0Z8R_hA6gVqVsy3Z5x9Yw1b0uIGSpE3XSkTUCjdzmng8SE9ydzCkcczaLFbV57M5b4vQbPwZR8EhMd6qDX-v_KoDlbWGIeFfp3KfGKYVbokMHRbm0bJgeggFJPwhjOb-lvKjojvPJDEBgd4cMlWml2I1KLLYhwN8C3DPJZqeTxfZwAgVpjxQQ5ScYqPiG7kTcM-te8vmUDsTMWX5vXjDlJKyhyIOMu5OfO3mqo3-No40zG8aNMQulUTX_XCVE8vSbIti5Qs87VjhS70ZAaSXo8QK5TjKT9VerID7Bl4u1I1CMXMiYLdk4fZhydSG4lKvHamOYSAKg-RL0DpbAG6Y3c-ftPFL7DKKqpjAAUm4HS1_eW52CoE0q3_ON_Yk_gTrXwLSS8oOfaZ_48rAPoHEcLODezD-xA4SdffBFtIynM9YGR5QBmiKWyPTX_X_K_9MLkYS4vzP-UHp6HIJctoLoztD_71wVhP6eOpybKUtPSD-8dFLdef3wXjLCGJKO3mx0G07icrit56StANR0VzFLRlaBkwxekk-74ec1X8vVzCTARjh-t5P02HD7kR9Z_Od9LKZ5dUnG1pgVXKSV87cdInLuU1pVcVlbEENec-Mc6ES2UQu3SWG8qBkziFHUl0nyUb4_8R4FBRy-sepba-zoN_UveLrLbLUn1LDCuKMRsYYwOUUPYWffcHil4fuan1j9hGZZsUJ9Dt0LmiByp9Ng-zW1UFOzt8uy-cUnl9FkdStC2-AvGKZ-PquRnuDpZcqsJ0pkEWwsBi9byoKYBSAcy259cHPA6mgxGljuv8K0IhQtJwdf-mOcrHgKJmikSM4yN9uB5eOcyLQB7a98GckUPwQRSWojyI6-lVTaKQIDH7em-PerKqj7X-WOeNjH5FTlk7ChuI7TzifXFOtyefbkf_wGMfu9CN_A1IaWbuzuYE71m4Tn_V0QyYJz8Pkhj9HAFrv_zTRcYNnTP8gUW745txwFUPTW_0wqQkuMmtJ3VoM2xmgJ81F3QavEy4VGC22nGz2RwuN0YQwDmRLo3rpWkEk5L6vxxe-Ek4B_BW9B79_MBbe2lXG0Ac5x9qAadZb7z-AKpfF_4VFi5g4ge9poYGJSlKTbkOaNgdDgTz2dhA786tv5ZmUiC6Yom4J0aKrVRQgxtAZvQTbQ4rERku02ATTW3fbu-srzbRzDxVWfUFUsrpePFTDpK0XJIdBrEeTeMAQiD4jwiG1TqP-jDKYlmUByNjuxXeKaHqzm6IKIXmDKr4ld3_UaE_cbZJfszsW7u8ZyhZ-CJwnCPUDcLVnntj2BR3r0JGOweNAuIISBJks5r6jM4pWDrBhPZ4G9T4Q5eK4vImUvmYVhgoaCuEF7Ux6LuE3U8f5Ym2EQL8FfCyCNPhlY1BjxSRicdpudnqU2cxdmxqW7Tb41IZonVYy6E9oh93X6SwEH37kASoXvxOb3rJ7b7T6ZsjQTiScA7HJeBZfs174TP_Bq7tnFCm35t0lHmL61Wwtc-ob1m9JpUruiI0lZ1jaksCWEitbVgbpS6o6eo7w-xjr1hzosQ9_pUpxmjpzpfbixILHnH_y6dxXcruJCYWIEXkAYeUpfgVcnYX9oOTApQLyyVO6KhY3i3ovmzsyhbuBrBUsC8LJDs0nTmcgINi5M9Ss53Z4kEIxIHlJvErmYE-NjOrD6Sc1yHbKWsSqtmXpESdEXKipGY5thAvvNOf-Zta7TGmXdUwf_RTnrcDw3Zojrhzlz8UyuA2JnjDocoDs8y89G_prTWtZ6Yj2pvDhxyoTEZXrpihEydwYIILXSjXSMpOAFXM9dNa_4kq0bmTpXEm2VkNu_IqXljnau7k-CVvIHQoDo2sViXewmh6vPkuXlKZYVgEmswvf3Qd-hcQNlPxzcuIhSefXz2VZV_fvu0Y6f2BuyfnVHUOs219MBr-ruVkPaxfbbQEtlV0yE_o.vTYR3P2e8MVXamfAOvAjew' 
                
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
                time.sleep(5)

            message = t['message']
            
            # write message and keyword to file
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

    def run(self):        
        if HARVESTER_ON:
            # spawn harvester process to grab chatgpt texts
            p = threading.Thread(target = self.harvest_chatgpt)
            p.start()
        while True:        
            root.update_idletasks()
            root.update()         

            if not self.get_video_ready and not self.get_video_running:
                print("RUNNING GET VIDEO")
                self.get_video() #threaded
            
            if not self.is_inference_ready and not self.is_inference_running:
                print("RUNNING INFERENCE")
                self.run_inference_tacotron() #threaded

            if self.get_video_ready and self.is_inference_ready and not self.is_play_audio_running:       
                print("STARTING SHOW")          
                self.play_audio() # threaded
                time.sleep(2)
                self.play_main_video()      
            time.sleep(1)
            
    def play(self):
        #copy ready video into main video
        shutil.copyfile("prep_video.mp4", "main_video.mp4")
        
        #copy new wavs
        for i, p in enumerate(self.wav_list):
            shutil.copyfile(p, "wav_{}.wav".format(i))

        self.is_audio_playing = True

        time.sleep(3)

        for i, w in enumerate(self.wav_list):
            wave_obj = sa.WaveObject.from_wave_file("wav_{}.wav".format(i))            
            play_obj = wave_obj.play()
            play_obj.wait_done()

        self.is_audio_playing = False
        self.is_inference_ready = False
        self.is_inference_running = False
        self.get_video_ready = False
        self.get_video_running = False
        self.is_play_audio_running = False

    def play_audio(self):
        if self.next_episode_label:
            if self.next_episode_label.winfo_ismapped():
                self.next_episode_label.pack_forget()
        self.is_play_audio_running = True
        self.t_play = threading.Thread(target=self.play, args=())
        self.t_play.start()

    def run_inference_tacotron(self):
        self.is_inference_running = True
        self.t_inf = threading.Thread(target=self.run_inference_tacotron_t, args=())
        self.t_inf.start()

    def get_video(self):     
        subdir = f"{PROJECT_PATH}_texts"      
        i = 0
        while True:
            filename = "{}/{}.txt".format(subdir, str(i))
            if not os.path.exists(filename):
                break
            i += 1
        r = random.randint(0, i)
        f = "{}/{}.txt".format(subdir, r)
        print("RUNNING VIDEO SEARCH FROM FILE {}".format(f))
        with open(f, 'r') as f:
            self.keyword = f.readline().strip()
            self.chatgpt_message = f.read().strip()

        self.get_video_running = True
        self.t_get_video = threading.Thread(target=self.get_video_link_and_download_t, args=())
        self.t_get_video.start()        
   
    def get_video_link_and_download_t(self):

        if os.path.exists("prep_video.mp4"):
            os.remove('prep_video.mp4')
        if os.path.exists("prep_video.part.mp4"):
            os.remove('prep_video.part.mp4')            
        
        while not os.path.exists("prep_video.mp4"):
            while True:
                try:
                    videosSearch = None
                    while not videosSearch:
                        videosSearch = VideosSearch(self.keyword, limit = 10) # limit = max number of returned videos
                        time.sleep(0.5)       

                    # check durations and find best
                    
                    results = videosSearch.result()
                    filtered = []
                    for x in results['result']:
                        if not "none" in x['duration']:
                            d = x['duration'].split(':')[0]
                            if int(d) <= 10 and int(d) >= 3:    
                                filtered.append(x)

                    r = random.randint(0,len(filtered)-1)
                except:
                    print("Video search failed, trying again with different keyword!")
                    self.keyword = "nature {}".format(self.keyword)
                    time.sleep(0.5)
                else:
                    break

            print("VIDEO SEARCH DONE")

            #get video title
            self.video_title = filtered[r]['title']

            link =  filtered[r]['link'] 
            
            # os.system("yt-dlp {} -f 137 -o main_video.%(ext)s".format(link))
            # os.system("yt-dlp {} -f 136 -o prep_video.%(ext)s".format(link))
            self.is_download_video_running = True
            os.system("yt-dlp {} -f bv*[ext=mp4] --max-filesize 500M -o prep_video.%(ext)s".format(link))
            self.is_download_video_running = False
            time.sleep(1)

        self.get_video_ready = True
        self.get_video_running = False

    def play_main_video(self):
        vlcmedia = self.vlc_obj.media_new("main_video.mp4")
        self.vlcplayer.set_media(vlcmedia)
        self.vlcplayer.set_hwnd(video_frame.winfo_id())#tkinter label or frame

        self.vlcplayer.audio_set_mute(True)
        self.vlcplayer.play()
        self.title_label.config(text="{}:: {}".format(self.keyword, self.video_title))
        self.title_label.pack(side="bottom", anchor="sw")
        video_title_timer = time.time()

        playing = set([1,2,3,4])
        play = True
        while play:
            root.update_idletasks()
            root.update()
            if time.time() - video_title_timer >= 10:
                if self.title_label.winfo_ismapped():
                    self.title_label.pack_forget()
            time.sleep(1)
            if self.is_audio_playing == False:
                play = False
            state = self.vlcplayer.get_state()
            if state in playing:
                continue
            else:
                play = False
        print("PLAYING EPISODE DONE")        
        self.next_episode_label.pack(side="bottom", anchor="se")
    
    def play_bumper_video(self):
        pass

    def run_inference_tacotron_t(self):       
        # make 2 lines into one for better inference       
        #Split text 
        text = self.chatgpt_message
        text = text.strip('\n')
        text = text.strip('"')
        text = text.strip('”')
        text = text.strip('“')            
        text_list = re.split(r'(?<=[\.\!\?])\s*', text)

        #remove blank and short cuts
        text_list_cleaned = []
        x = 0
        joined_text = ""
        for i in text_list:
            if i and len(i) > 10:
                if x == 0:
                    joined_text = i
                else:
                    joined_text = "{} {}".format(joined_text, i)
                    text_list_cleaned.append(joined_text)
                    joined_text = ""
                    x = 0
                x += 1
        if joined_text:
            text_list_cleaned.append(joined_text)

        # text_list = [i for i in text_list if i]
        # print(text_list_cleaned)
        wav_list = []

        for i, t in enumerate(text_list_cleaned):
            os.makedirs("wav_out", exist_ok=True)
            wav_file = "wav_out/out{}.wav".format(i)        
            sequence = np.array(text_to_sequence(t, ['english_cleaners']))[None, :]
            sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
            mel_outputs, mel_outputs_postnet, _, alignments = self.taco_model.inference(sequence)    
            
            print("Using waveglow model")
            for k in self.waveglow_model.convinv:
                k.float()
            self.denoiser = Denoiser(self.waveglow_model) 
            with torch.no_grad():
                audio = self.waveglow_model.infer(mel_outputs_postnet, sigma=1)
            audio_denoised = self.denoiser(audio, strength=0.02)[:, 0]
            audioout = audio_denoised[0].data.cpu().numpy()
            #audioout = audio[0].data.cpu().numpy()
            audioout32 = np.float32(audioout)
            sf.write(wav_file, audioout32, 22050)
            wav_list.append(wav_file)
            time.sleep(0.5)
        self.wav_list = wav_list            
        self.is_inference_ready = True
        self.is_inference_running = False
            
# Create tkinter window
root = tk.Tk()
root.title("VLC player in Tkinter")
root.geometry("1280x720")

video_frame = tk.Frame(root, bg="black")
video_frame.pack(fill=tk.BOTH, expand=True)

control = Control()
control.run()
root.mainloop()

