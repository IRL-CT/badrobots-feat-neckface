import os
import subprocess
from tqdm import tqdm

class ConvertVideo:
    def __init__(self, data_path, to_type="mp4"):
        self.files_to_ignore = [".DS_Store"]
        self.to_type = to_type
        self.data_path = data_path
    
    def to_mp4(self, video_path):
        output_path = os.path.join(self.data_path, f"neckface_dataset/neckface_device_recordings/format_{self.to_type}/")
        os.makedirs(output_path, exist_ok=True)

        videos = [
            video for video in os.listdir(video_path)
            if video not in self.files_to_ignore and video.endswith(".avi")
        ]
        
        for video in tqdm(videos, total=len(videos), desc="Processing Videos: "):
            video_name = os.path.splitext(video)[0]
            original_video_path = os.path.join(video_path, video)
            converted_video_path = os.path.join(output_path, f"{video_name}.{self.to_type}")
            
            try:
                command = [
                    "ffmpeg",
                    "-i", original_video_path,
                    "-vcodec", "libx264",
                    "-acodec", "aac",
                    converted_video_path
                ]
                subprocess.run(command)
                print(f"Successfully converted {video_name} to {self.to_type}")
            except Exception as e:
                print(f"Unable to process converting file: {video} Error: {e}")

if __name__ == "__main__":
    data_path = "../../data/"
    avi_path = data_path + "../../data/neckface_dataset/neckface_device_recordings/format_avi/"
    conversion_obj = ConvertVideo(data_path=data_path)
    conversion_obj.to_mp4(video_path=avi_path)
