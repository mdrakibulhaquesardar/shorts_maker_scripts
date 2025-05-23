# YouTube Shorts Maker

এই Python script টি YouTube ভিডিও থেকে automatically shorts তৈরি করে। এটি ভিডিও থেকে best clips খুঁজে বের করে, copyright-free করে এবং shorts format এ export করে।

## Features

- YouTube ভিডিও থেকে automatically best clips খুঁজে বের করে
- Scene detection এবং dialogue detection ব্যবহার করে best moments খুঁজে বের করে
- ভিডিও quality 720p এ optimize করে
- Copyright-free করার জন্য multiple transformations apply করে:
  - Mirror effect
  - Speed change
  - Color grading
  - Crop
  - Text overlay
  - Audio pitch shift
- Smooth transitions between clips
- Beautiful loading animations
- Colored console output

## Requirements

### System Requirements
1. Python 3.7+ installed
2. FFmpeg installed
3. ImageMagick installed (for text overlay)

### Python Packages
Script টি run করার জন্য নিচের packages গুলো install করতে হবে:

#### Core Packages
- yt-dlp: YouTube video download
- moviepy: Video editing
- opencv-python: Image processing
- numpy: Numerical operations
- scipy: Scientific computing
- librosa: Audio processing
- scikit-learn: Machine learning
- tqdm: Progress bars
- colorama: Colored console output
- ffmpeg-python: FFmpeg integration

## Installation

1. প্রথমে Python install করুন: [Python Download](https://www.python.org/downloads/)

2. FFmpeg install করুন:
   - Windows: [FFmpeg Download](https://ffmpeg.org/download.html)
   - Linux: `sudo apt-get install ffmpeg`
   - Mac: `brew install ffmpeg`

3. ImageMagick install করুন:
   - Windows: [ImageMagick Download](https://imagemagick.org/script/download.php#windows)
   - Linux: `sudo apt-get install imagemagick`
   - Mac: `brew install imagemagick`

4. Virtual Environment Setup:
```bash
# Windows এ:
# 1. Project folder এ যান
cd path\to\shorts_maker_scripts

# 2. Virtual environment create করুন
python -m venv venv

# 3. Virtual environment activate করুন
venv\Scripts\activate

# 4. pip update করুন
python -m pip install --upgrade pip

# 5. Required packages install করুন
pip install -r requirements.txt

# Linux/Mac এ:
# 1. Project folder এ যান
cd path/to/shorts_maker_scripts

# 2. Virtual environment create করুন
python3 -m venv venv

# 3. Virtual environment activate করুন
source venv/bin/activate

# 4. pip update করুন
python -m pip install --upgrade pip

# 5. Required packages install করুন
pip install -r requirements.txt
```

5. Virtual Environment Deactivate:
```bash
# Windows/Linux/Mac এ:
deactivate
```

6. Virtual Environment Reuse:
```bash
# Windows এ:
venv\Scripts\activate

# Linux/Mac এ:
source venv/bin/activate
```

## How to Use

1. Virtual environment activate করুন:
```bash
# Windows এ:
venv\Scripts\activate

# Linux/Mac এ:
source venv/bin/activate
```

2. Script টি run করুন:
```bash
python main.py
```

3. YouTube video URL দিন

4. Video duration দিন (seconds এ, default 120s)

5. Number of clips দিন (default 3)

6. Script automatically:
   - ভিডিও download করবে
   - Best clips খুঁজে বের করবে
   - Transformations apply করবে
   - Final video save করবে

## Output

- Processed videos `processed_videos` folder এ save হবে
- Filename format: `processed_video_YYYYMMDD_HHMMSS.mp4`
- Video quality: 720p
- Format: MP4

## Troubleshooting

1. FFmpeg error:
   - FFmpeg properly installed আছে কিনা check করুন
   - System PATH এ FFmpeg add করা আছে কিনা check করুন

2. ImageMagick error:
   - ImageMagick properly installed আছে কিনা check করুন
   - Text overlay ছাড়া script run হবে

3. Download error:
   - Internet connection check করুন
   - Video URL valid কিনা check করুন

4. Processing error:
   - Sufficient disk space আছে কিনা check করুন
   - Video format supported কিনা check করুন

5. Package installation error:
   - Python version check করুন (3.7+ required)
   - pip update করুন: `python -m pip install --upgrade pip`
   - Virtual environment use করুন
   - System dependencies install করুন

6. Virtual Environment error:
   - Python venv module installed আছে কিনা check করুন
   - Windows এ: `python -m pip install virtualenv`
   - Linux/Mac এ: `sudo apt-get install python3-venv`
   - Virtual environment properly activated আছে কিনা check করুন
   - Terminal এ (venv) prefix দেখতে পাবেন

## Notes

- Processing time video length এবং system performance এর উপর depend করে
- Longer videos বেশি time নিতে পারে
- Better quality videos বেশি disk space use করবে
- Virtual environment use করা recommended
- Virtual environment activate না করে script run করবেন না
- Project folder এ .gitignore file add করুন এবং venv/ folder ignore করুন

## Support

যদি কোন problem হয়:
1. Error message read করুন
2. Troubleshooting section check করুন
3. Issue report করুন

## License

MIT License 