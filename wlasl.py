import json
from pathlib import Path
import subprocess, os
from tqdm import tqdm

target_glosses = [
  "hello","thank you","please","yes","no","eat","drink","like","want",
  "who","what","where","when","why","sleep","help","good","bad","see",
  "again","understand","me","you","milk"
]


meta = json.load(open("WLASL_v0.3.json", "r")) 
target_set = set(target_glosses)

filtered = []
for item in meta:
    if item.get("gloss") in target_set:
        for inst in item.get("instances", []):
            filtered.append({
                "gloss": item["gloss"],
                "video_id": inst.get("video_id", inst.get("youtube_id")),
                "url": inst["url"],                      # ✅ now inside instance
                "start_frame": inst.get("frame_start", 0),
                "end_frame": inst.get("frame_end", None)
            })

print(f"Found {len(filtered)} clips for your selected glosses")




out_root = Path("videos")
out_root.mkdir(exist_ok=True)
for item in tqdm(filtered):
    gloss = item["gloss"]
    url = item["url"]
    vid = item["video_id"]
    outdir = out_root / gloss
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{vid}.mp4"
    if outpath.exists():
        continue
    # yt-dlp command
    cmd = ["yt-dlp", "-f", "mp4", "-o", str(outpath), url]
    subprocess.run(cmd)   # add error handling / retries in production
