from video_processor import VideoProcessor
from utils import print_success, print_warning, print_error, print_info
from colorama import Fore, Style

def main():
    processor = VideoProcessor()
    
    # Get YouTube URL from user
    url = input(f"{Fore.CYAN}Enter YouTube video URL: {Style.RESET_ALL}")
    
    # Get video duration from user
    while True:
        try:
            duration = input(f"{Fore.CYAN}Enter video duration in seconds (press Enter for default 120s): {Style.RESET_ALL}")
            if duration.strip() == "":
                duration = 120
            else:
                duration = float(duration)
                if duration <= 0:
                    print_warning("Duration must be greater than 0")
                    continue
            break
        except ValueError:
            print_error("Please enter a valid number")
    
    # Get number of clips from user
    while True:
        try:
            num_clips = input(f"{Fore.CYAN}Enter number of clips to combine (press Enter for default 3): {Style.RESET_ALL}")
            if num_clips.strip() == "":
                num_clips = 3
            else:
                num_clips = int(num_clips)
                if num_clips <= 0:
                    print_warning("Number of clips must be greater than 0")
                    continue
            break
        except ValueError:
            print_error("Please enter a valid number")
    
    # Download the video
    video_path = processor.download_youtube_video(url)
    if not video_path:
        return
    
    # Find the best clips
    clips = processor.find_best_clips(video_path, target_duration=duration, num_clips=num_clips)
    print_info(f"\nSelected {len(clips)} clips:")
    for i, (start, end) in enumerate(clips, 1):
        print_info(f"Clip {i}: {start:.2f}s to {end:.2f}s")
    
    # Process the video
    output_path = processor.apply_transformations(video_path, clips)
    if output_path:
        print_success("\nVideo processing completed successfully!")
        print_info(f"Your processed video is saved at: {output_path}")

if __name__ == "__main__":
    main() 