import os
# Try different possible ImageMagick paths
possible_paths = [
    r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
    r"C:\Program Files (x86)\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
    r"C:\Program Files\ImageMagick-7.1.1\magick.exe",
    r"C:\Program Files (x86)\ImageMagick-7.1.1\magick.exe"
]

magick_path = None
for path in possible_paths:
    if os.path.exists(path):
        magick_path = path
        break

if magick_path:
    os.environ["IMAGEMAGICK_BINARY"] = magick_path
else:
    print("Warning: ImageMagick not found. Text overlay might not work.")
    print("Please install ImageMagick from: https://imagemagick.org/script/download.php#windows")

import yt_dlp
from moviepy.editor import VideoFileClip, vfx, CompositeVideoClip, AudioFileClip, TextClip, concatenate_videoclips
import cv2
import numpy as np
from datetime import datetime
from scipy.io import wavfile
import librosa
from sklearn.cluster import KMeans
import re
import time
from tqdm import tqdm
import threading
import sys
from colorama import init, Fore, Back, Style
from utils import (
    print_loading, start_loading, stop_loading,
    print_success, print_warning, print_error, print_info,
    show_loading_animation
)

# Initialize colorama
init()

def print_loading(message):
    """Print loading animation with color"""
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    while not hasattr(print_loading, "stop"):
        sys.stdout.write(f"\r{Fore.CYAN}{message} {chars[i]}{Style.RESET_ALL}")
        sys.stdout.flush()
        time.sleep(0.1)
        i = (i + 1) % len(chars)
    sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")
    sys.stdout.flush()

def start_loading(message):
    """Start loading animation in a separate thread"""
    print_loading.stop = False
    thread = threading.Thread(target=print_loading, args=(message,))
    thread.daemon = True
    thread.start()
    return thread

def stop_loading(thread):
    """Stop loading animation"""
    print_loading.stop = True
    thread.join()

def print_success(message):
    """Print success message in green"""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")

def print_warning(message):
    """Print warning message in yellow"""
    print(f"{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}")

def print_error(message):
    """Print error message in red"""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")

def print_info(message):
    """Print info message in blue"""
    print(f"{Fore.BLUE}ℹ {message}{Style.RESET_ALL}")

class VideoProcessor:
    def __init__(self):
        self.output_dir = "processed_videos"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print_success(f"Created output directory: {self.output_dir}")

    def sanitize_filename(self, filename):
        """Convert filename to a safe format"""
        # Remove special characters and replace spaces with underscores
        safe_name = re.sub(r'[^\w\-_.]', '_', filename)
        # Ensure the filename isn't too long
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        return safe_name

    def download_youtube_video(self, url):
        """Download video from YouTube using yt-dlp"""
        try:
            loading_thread = start_loading("Downloading video")
            
            ydl_opts = {
                'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best[ext=mp4]',
                'outtmpl': os.path.join(self.output_dir, '%(id)s.%(ext)s'),
                'quiet': False,
                'no_warnings': False,
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',
                }],
                'merge_output_format': 'mp4',
                'progress_hooks': [lambda d: print(f"\r{Fore.CYAN}Downloading: {d['_percent_str']} of {d['_total_bytes_str']}{Style.RESET_ALL}", end='') if d['status'] == 'downloading' else None]
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_id = info['id']
                video_path = os.path.join(self.output_dir, f"{video_id}.mp4")
                print_success(f"Downloaded successfully to: {video_path}")
                return video_path
                
        except Exception as e:
            print_error(f"Error downloading video: {str(e)}")
            print_warning("Please try a different video URL or check your internet connection")
            return None
        finally:
            stop_loading(loading_thread)

    def detect_scenes(self, video_path, threshold=30.0):
        """Detect scene changes in the video"""
        print_info("Detecting scenes...")
        loading_thread = start_loading("Analyzing video frames")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print_error(f"Could not open video file: {video_path}")
                return []
                
            scenes = []
            prev_frame = None
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Process every 5th frame for speed
            frame_interval = 5
            
            with tqdm(total=total_frames//frame_interval, desc="Scene Detection", unit="frames") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    if frame_count % frame_interval == 0:
                        if prev_frame is not None:
                            # Resize frames for faster comparison
                            frame = cv2.resize(frame, (320, 180))
                            prev_frame = cv2.resize(prev_frame, (320, 180))
                            
                            # Calculate difference between frames
                            diff = cv2.absdiff(frame, prev_frame)
                            non_zero_count = np.count_nonzero(diff)
                            
                            if non_zero_count > threshold * frame.size:
                                scenes.append(frame_count / fps)
                        
                        prev_frame = frame
                        pbar.update(1)
                    
                    frame_count += 1
                    
            cap.release()
            print_success(f"Detected {len(scenes)} scenes")
            return scenes
            
        finally:
            stop_loading(loading_thread)

    def detect_dialogue_segments(self, video_path, min_silence_len=1000, silence_thresh=-40):
        """Detect segments with dialogue using audio analysis"""
        print_info("Detecting dialogue segments...")
        loading_thread = start_loading("Analyzing audio")
        
        try:
            # Extract audio
            video = VideoFileClip(video_path)
            if video.audio is None:
                print_warning("No audio track found in video")
                return []
                
            audio = video.audio
            audio_path = os.path.join(self.output_dir, "temp_audio.wav")
            
            # Show progress for audio extraction
            print_info("Extracting audio...")
            audio.write_audiofile(audio_path, verbose=False, logger=None)
            
            # Load audio file with lower sample rate for speed
            print_info("Loading audio...")
            y, sr = librosa.load(audio_path, sr=22050)
            
            # Detect non-silent chunks
            print_info("Detecting dialogue...")
            non_silent_ranges = librosa.effects.split(y, 
                                                    top_db=abs(silence_thresh),
                                                    frame_length=min_silence_len)
            
            # Convert to seconds
            dialogue_segments = [(start/sr, end/sr) for start, end in non_silent_ranges]
            
            # Cleanup
            os.remove(audio_path)
            video.close()
            
            print_success(f"Detected {len(dialogue_segments)} dialogue segments")
            return dialogue_segments
            
        except Exception as e:
            print_error(f"Error in dialogue detection: {str(e)}")
            return []
        finally:
            stop_loading(loading_thread)

    def find_best_clips(self, video_path, target_duration=120, num_clips=3):
        """Find multiple best clips based on scene and dialogue detection"""
        print_info("Finding best clips...")
        scenes = self.detect_scenes(video_path)
        dialogue_segments = self.detect_dialogue_segments(video_path)
        
        if not scenes and not dialogue_segments:
            print_warning("Could not detect scenes or dialogue. Using default clips.")
            return [(0, target_duration/num_clips)]
        
        # Combine scene changes and dialogue segments
        all_points = sorted(list(set(scenes + [s for seg in dialogue_segments for s in seg])))
        
        if not all_points:
            return [(0, target_duration/num_clips)]
            
        # Find segments with most activity
        clip_scores = []
        clip_duration = target_duration / num_clips
        video = VideoFileClip(video_path)
        total_duration = video.duration
        video.close()
        
        # Divide video into sections
        section_duration = total_duration / num_clips
        
        for section in range(num_clips):
            section_start = section * section_duration
            section_end = (section + 1) * section_duration
            
            # Find points in this section
            section_points = [p for p in all_points if section_start <= p <= section_end]
            
            if not section_points:
                # If no points in section, use middle of section
                mid_point = (section_start + section_end) / 2
                clip_scores.append((mid_point, mid_point + clip_duration, 0))
                continue
            
            # Score each possible clip in this section
            for start in section_points:
                end = start + clip_duration
                if end > section_end:
                    continue
                
                # Count scene changes and dialogue in this segment
                score = sum(1 for s in scenes if start <= s <= end)
                score += sum(1 for seg in dialogue_segments if start <= seg[0] <= end or start <= seg[1] <= end)
                
                clip_scores.append((start, end, score))
        
        # Sort by score and get best clip from each section
        clip_scores.sort(key=lambda x: x[2], reverse=True)
        best_clips = []
        used_sections = set()
        
        for start, end, score in clip_scores:
            # Find which section this clip belongs to
            section = int(start / section_duration)
            
            # If we haven't used this section yet, add the clip
            if section not in used_sections:
                best_clips.append((start, end))
                used_sections.add(section)
                
                # If we have enough clips, stop
                if len(best_clips) == num_clips:
                    break
        
        # Sort clips by start time
        best_clips.sort(key=lambda x: x[0])
        
        print_success(f"Selected {len(best_clips)} clips")
        return best_clips

    def apply_transformations(self, video_path, clips):
        """Apply transformations to make video copyright-free and export in 720p"""
        try:
            processed_clips = []
            video = VideoFileClip(video_path)
            
            for i, (start_time, end_time) in enumerate(clips, 1):
                loading_thread = start_loading(f"Processing clip {i}/{len(clips)}")
                
                try:
                    # Get subclip
                    clip = video.subclip(start_time, end_time)
                    
                    # 1. Mirror effect
                    clip = clip.fx(vfx.mirror_x)
                    
                    # 2. Slight speed change
                    clip = clip.fx(vfx.speedx, 1.05)
                    
                    # 3. Subtle color grading with CLAHE
                    def color_transform(frame):
                        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        cl = clahe.apply(l)
                        limg = cv2.merge((cl,a,b))
                        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
                        return enhanced
                    clip = clip.fl_image(color_transform)
                    
                    # 4. Crop a few pixels from each side
                    w, h = clip.size
                    crop_x = int(w * 0.05)
                    crop_y = int(h * 0.05)
                    clip = clip.crop(x1=crop_x, y1=crop_y, x2=w-crop_x, y2=h-crop_y)
                    
                    # 5. Add a text overlay
                    try:
                        txt = TextClip("ShortsMaker", fontsize=30, color='grey', font='Arial-Bold', size=(1280, 720))
                        txt = txt.set_position(('left', 'bottom'))
                        txt = txt.set_duration(clip.duration).set_opacity(0.7)
                        clip = CompositeVideoClip([clip, txt])
                    except Exception as e:
                        print_warning(f"Could not add text overlay: {str(e)}")
                        print_info("Continuing without text overlay...")
                    
                    # 6. Resize to 720p
                    clip = clip.resize((1280, 720))
                    
                    # 7. Pitch shift the audio
                    if clip.audio is not None:
                        try:
                            # Start loading indicator for audio processing
                            audio_loading_thread = start_loading(f"Processing audio for clip {i}")
                            
                            try:
                                # Extract audio
                                audio = clip.audio
                                audio_path = os.path.join(self.output_dir, f"temp_audio_{i}.wav")
                                
                                # Extract audio with progress
                                print_info(f"Extracting audio for clip {i}...")
                                audio.write_audiofile(audio_path, verbose=False, logger=None)
                                
                                # Load and process audio with progress
                                print_info(f"Loading audio for clip {i}...")
                                y, sr = librosa.load(audio_path, sr=None)
                                
                                print_info(f"Pitch shifting audio for clip {i}...")
                                y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
                                
                                shifted_audio_path = os.path.join(self.output_dir, f"temp_audio_shifted_{i}.wav")
                                wavfile.write(shifted_audio_path, sr, (y_shifted * 32767).astype(np.int16))
                                
                                print_info(f"Applying processed audio to clip {i}...")
                                new_audio = AudioFileClip(shifted_audio_path)
                                new_audio = new_audio.set_duration(clip.duration)
                                
                                clip = clip.set_audio(new_audio)
                                
                                # Cleanup
                                os.remove(audio_path)
                                os.remove(shifted_audio_path)
                                print_success(f"Audio processing completed for clip {i}")
                                
                            finally:
                                stop_loading(audio_loading_thread)
                                
                        except Exception as e:
                            print_warning(f"Could not process audio for clip {i}: {str(e)}")
                            # Start loading indicator for original audio
                            original_audio_thread = start_loading(f"Using original audio for clip {i}")
                            try:
                                # Keep original audio
                                print_info(f"Keeping original audio for clip {i}...")
                                time.sleep(1)  # Show loading for a moment
                            finally:
                                stop_loading(original_audio_thread)
                            
                            # Show animated loading message
                            show_loading_animation("Continuing with original audio", duration=3, color=Fore.CYAN)
                            print_info("Original audio applied successfully")
                    
                    processed_clips.append(clip)
                    
                finally:
                    stop_loading(loading_thread)
            
            # Combine all clips with smooth transitions
            loading_thread = start_loading("Combining clips")
            print_success("Combining All clips...")
            
            try:
                # Add crossfade between clips
                crossfade_duration = 1.0  # 1 second crossfade
                final_clips = []
                
                for i, clip in enumerate(processed_clips):
                    if i > 0:  # Add crossfade for all clips except the first one
                        # Create crossfade effect
                        clip = clip.crossfadein(crossfade_duration)
                    final_clips.append(clip)
                
                final_video = concatenate_videoclips(final_clips, method="compose")
                
                # Output with 720p settings
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(self.output_dir, f"processed_video_{timestamp}.mp4")
                
                # Start loading indicator for video export
                export_loading_thread = start_loading("Exporting video")
                
                try:
                    final_video.write_videofile(
                        output_path,
                        codec='libx264',
                        audio_codec='aac',
                        bitrate="4000k",
                        threads=4,
                        preset='medium',
                        ffmpeg_params=[
                            '-crf', '23',
                            '-profile:v', 'main',
                            '-level', '3.1',
                            '-pix_fmt', 'yuv420p',
                            '-movflags', '+faststart'
                        ],
                        verbose=False,
                        logger=None
                    )
                finally:
                    stop_loading(export_loading_thread)
                
                # Cleanup
                video.close()
                final_video.close()
                for clip in processed_clips:
                    clip.close()
                
                # Show final processing animation
                show_loading_animation("Finalizing video", duration=2, color=Fore.GREEN)
                
                print_success(f"Processed video saved to: {output_path}")
                return output_path
                
            finally:
                stop_loading(loading_thread)
            
        except Exception as e:
            print_error(f"Error processing video: {str(e)}")
            return None

    # ... [rest of the VideoProcessor class remains the same] ... 