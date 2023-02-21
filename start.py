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

TEXTS_PATH = "attenborough2_texts"
EPISODES_PATH = "da_episodes"

MODEL_TYPE = "VITS" #VITS or TACO

VITS_MODEL_PATH = "da_checkpoint.pth"
VITS_CONFIG_PATH = "da_config.json"
MODEL_PREFIX = "da_"

class Control:
    def __init__(self):
        self.first_run = True
        self.is_episode_running = False
        self.run_inference = False
        self.is_audio_playing = False
        self.is_scene_running = False
        self.is_inference_ready = False
        self.get_video_ready = False
        self.get_video_running = False
        self.is_play_audio_running = False
        self.vlc_obj = vlc.Instance("--no-xlib")
        self.vlc_audio_obj = vlc.Instance("--no-xlib")        
        self.vlcplayer = self.vlc_obj.media_player_new()
        self.vlcplayer_audio = self.vlc_audio_obj.media_player_new()
        
        self.next_episode_label = tk.Label(root, text="Generating next episode", font=("TkDefaultFont", 20), width=20, bg='black', fg='white')        
        self.title_label = tk.Label(root, text="", font=("TkDefaultFont", 15), bg='black', fg='white')
        self.video_title = ""
        self.video_link = ""

        self.running_keyword = ""
        self.running_video_path = ""
        self.running_video_title = ""
        self.running_chatgpt_message = ""
        
        self.file_number = ""
        self.keyword = ""
        self.chatgpt_message = ""        
        self.session_token = None        
        self.is_download_video_running = False
        self.playlist = []

    def run(self):        

        self.first_run = True
        print("RUNNING GET VIDEO")
        self.get_video() #threaded
        while True:        
            root.update_idletasks()
            root.update()        
            
            if self.first_run:
                if MODEL_TYPE == "VITS":
                    self.run_inference_vits_t()

            if not self.is_play_audio_running and self.is_inference_ready:      
                
                print("STARTING SHOW")

                self.vlcplayer_audio.set_hwnd(video_frame_audio.winfo_id())#tkinter label or frame    
                vlcmedia_audio = self.vlc_audio_obj.media_new(f"{MODEL_PREFIX}speech.wav")
                self.vlcplayer_audio.set_media(vlcmedia_audio)                

                self.play_audio() # threaded
                time.sleep(2)
                self.play_main_video()    
                self.vlcplayer.stop()
                print(f"FILENUM: {self.file_number} RUNVIDPATH: {self.running_video_path}")
                
                os.remove(f"{EPISODES_PATH}/{self.file_number-1}.mp4")

                time.sleep(1)
                self.first_run = False

            
    def play(self):

        self.is_episode_running = True
        if MODEL_TYPE == "VITS":
            self.is_audio_playing = True
            time.sleep(2)
            self.vlcplayer_audio.play()
            self.vlcplayer_audio.audio_set_mute(False)      

            while self.vlcplayer_audio.get_state() == vlc.State.Playing or self.vlcplayer_audio.get_state() == vlc.State.NothingSpecial:
                time.sleep(1)

            print("PLAYING AUDIO DONE")

        self.is_audio_playing = False
        self.run_inference_vits_t()        
        
        self.is_episode_running = False
        self.is_play_audio_running = False

    def play_audio(self):
        if self.next_episode_label:
            if self.next_episode_label.winfo_ismapped():
                self.next_episode_label.pack_forget()
        self.is_play_audio_running = True
        self.t_play = threading.Thread(target=self.play, args=())
        self.t_play.start()

    def get_video(self):        

        self.get_video_running = True
        self.t_get_video = threading.Thread(target=self.get_video_link_and_download_t, args=())
        self.t_get_video.start()        

    def play_main_video(self):    

        print(f"Playing video {self.running_video_path}")   

        vlcmedia = self.vlc_obj.media_new(self.running_video_path)
        self.vlcplayer.set_media(vlcmedia)
        self.vlcplayer.set_hwnd(video_frame.winfo_id())#tkinter label or frame

        self.vlcplayer.audio_set_mute(True)
        self.vlcplayer.play()
        self.title_label.config(text="{}:: {}".format(self.running_keyword, self.running_video_title))
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
                self.next_episode_label.pack(side="bottom", anchor="se")
                play = False
            state = self.vlcplayer.get_state()
            if state in playing:
                continue
            else:
                play = False
        print("PLAYING VIDEO DONE")        

        while self.is_episode_running:
            root.update_idletasks()
            root.update()
            time.sleep(1)
   
    def run_inference_vits_t(self):

        while not self.playlist:
            print("WAITING FOR PLAYLIST TO BE READY")
            time.sleep(5)
        self.file_number = self.playlist.pop(0)    

        with open(f"{EPISODES_PATH}/{self.file_number}.txt", 'r') as f:
            self.running_keyword = f.readline().strip()
            self.running_video_title = f.readline().strip()
            self.running_chatgpt_message = f.read()
        self.running_video_path = f"{EPISODES_PATH}/{self.file_number}.mp4"

        os.remove(f"{EPISODES_PATH}/{self.file_number}.txt")
      
        print(f"NEXT SHOW KEYWORD: {self.running_keyword} from file {self.file_number}.txt")       

        text = self.running_chatgpt_message
        text = text.replace('\n', ' ') 
        pattern = r'[\w\s\.,:\-\'!?]+'
        text = ''.join(re.findall(pattern, text))
        OUTPUT_PATH = f"{MODEL_PREFIX}speech.wav"
        if os.path.exists(OUTPUT_PATH):
            os.remove(OUTPUT_PATH)
        cmd = f'tts --text "{text}" --model_path {VITS_MODEL_PATH} --config_path {VITS_CONFIG_PATH} --out_path {OUTPUT_PATH} --use_cuda USE_CUDA'
        # print(cmd)

        os.system(cmd)
        print("DONE RUNNING INFERENCE")
        self.is_inference_ready = True
        self.run_inference = False
   
    def get_video_link_and_download_t(self):

        # clear old folder
        if os.path.exists(EPISODES_PATH):
            shutil.rmtree(EPISODES_PATH)
       
        os.mkdir(EPISODES_PATH)

        file_number = 0

        while True:    
            while len(self.playlist) > 10:
                print("WAITING FOR PLAYLIST TO CLEAR")
                time.sleep(60)

            files = os.listdir(TEXTS_PATH)
            random_file = random.choice(files)

            print("PREPPING EPISODE {}".format(random_file))
            with open(f"{TEXTS_PATH}/{random_file}", 'r') as f:
                self.keyword = f.readline().strip()
                self.chatgpt_message = f.read().strip()
        
            while True:
                try:
                    videosSearch = None
                    while not videosSearch:
                        videosSearch = VideosSearch(f"{self.keyword} in nature", limit = 10) # limit = max number of returned videos
                        time.sleep(0.5)       

                    # check durations and find best
                    
                    results = videosSearch.result()
                    filtered = []
                    for x in results['result']:
                        if not "none" in x['duration']:
                            d = x['duration'].split(':')[0]
                            if int(d) <= 20 and int(d) >= 3:    
                                filtered.append(x)

                    r = random.randint(0,len(filtered)-1)
                except:
                    print("Video search failed, trying again with different keyword!")
                    self.keyword = f"{self.keyword} facts"
                    time.sleep(0.5)
                else:
                    break

            #get video title
            self.video_title = filtered[r]['title']
            link =  filtered[r]['link']             
            # os.system("yt-dlp {} -f 137 -o main_video.%(ext)s".format(link))
            # os.system("yt-dlp {} -f 136 -o prep_video.%(ext)s".format(link))
            os.system(f'yt-dlp {link} -N 100 --quiet --download-sections "*0:10-3:40" -f bv*[ext=mp4] --max-filesize 500M -o {EPISODES_PATH}/{file_number}.%(ext)s')

            # write info to info file and add to playlist
            with open(f"{EPISODES_PATH}/{file_number}.txt", 'w', encoding="utf-8") as f:
                f.write(f"{self.keyword}\n")   
                pattern = r'[\w\s\.,:\-\'!?]+'
                text = ''.join(re.findall(pattern, self.video_title))
                f.write(f"{text}\n")    
                f.write(f"{self.chatgpt_message}")

            self.playlist.append(file_number)
            file_number += 1

            # print(f"ADDED {link}")
            time.sleep(1)

            
# Create tkinter window
root = tk.Tk()
root_audio = tk.Tk()
root.title("DAVID ATTERNBOROUGH VLC player in Tkinter")
root.geometry("1280x720+1000+0")
root_audio.title("David Attenborough Voice")
root_audio.geometry("200x200+800+300")

video_frame = tk.Frame(root, bg="black")
video_frame.pack(fill=tk.BOTH, expand=True)

video_frame_audio = tk.Frame(root_audio, bg="black")
video_frame_audio.pack(fill=tk.BOTH, expand=True)

control = Control()
control.run()
root.mainloop()

