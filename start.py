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

PROJECT_PATH = "attenborough2_texts"
TACOTRON_MODEL_PATH = "attenborough_v4_checkpoint_4225700(5050500)"
WAVEGLOW_MODEL_PATH = "attenborough_v4_waveglow_2030200(3133000)1e-5"

MODEL_TYPE = "TACO" #VITS or TACO

VITS_MODEL_PATH = "checkpoint_60000.pth"
VITS_CONFIG_PATH = "config.json"

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
    
    def run(self):        

        while True:        
            root.update_idletasks()
            root.update()         

            if not self.get_video_ready and not self.get_video_running:
                print("RUNNING GET VIDEO")
                self.get_video() #threaded
            
            if not self.is_inference_ready and not self.is_inference_running:
                print("RUNNING INFERENCE")
                if MODEL_TYPE == "VITS":
                    self.run_inference_vits()
                else:
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
        
        if MODEL_TYPE == "VITS":
            self.is_audio_playing = True
            time.sleep(3)
            wave_obj = sa.WaveObject.from_wave_file("speech.wav")            
            play_obj = wave_obj.play()
            play_obj.wait_done()
        else:
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

    def run_inference_vits(self):
        self.is_inference_running = True
        self.t_inf = threading.Thread(target=self.run_inference_vits_t, args=())
        self.t_inf.start()

    def get_video(self):     
        subdir = f"{PROJECT_PATH}"      
        i = 0
        while True:
            filename = "{}/{}.txt".format(subdir, str(i))
            if not os.path.exists(filename):
                break
            i += 1
        r = random.randint(0, i)
        f = "{}/{}.txt".format(subdir, r-1)
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
                        videosSearch = VideosSearch("{} facts".format(self.keyword), limit = 10) # limit = max number of returned videos
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

    def run_inference_vits_t(self):
        text = self.chatgpt_message
        text = text.replace('\n', '')
        text = text.replace('"', '')
        text = text.replace('”', '')
        text = text.replace('“', '')         
        OUTPUT_PATH = "speech.wav"
        cmd = 'tts --text "{}" --model_path {} --config_path {} --out_path {}'.format(text, VITS_MODEL_PATH, VITS_CONFIG_PATH, OUTPUT_PATH)
        print(cmd)
        os.system(cmd)
        print("DONE RUNNING INFERENCE")
        self.is_inference_ready = True
        self.is_inference_running = False

    def run_inference_tacotron_t(self):       
        # make 2 lines into one for better inference       
        #Split text 
        text = self.chatgpt_message
        text = text.replace('\n', '')
        text = text.replace('"', '')
        text = text.replace('”', '')
        text = text.replace('“', '')            
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

