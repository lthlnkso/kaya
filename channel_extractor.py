#!/usr/bin/env python3
"""
YouTube Channel Video Frame Extractor

Downloads videos from a YouTube channel within a date range and extracts frames.
Designed for image classification tasks (e.g., dog recognition).
"""

import argparse
import os
import subprocess
import threading
import queue
import cv2
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import yt_dlp


# Resolution: 720p is good for dog classification - enough detail without excessive storage
# Video-only (no audio) since we only need frames
VIDEO_FORMAT = "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/best[height<=720]"


def get_channel_videos(channel_input: str, start_date: datetime, end_date: datetime) -> list[dict]:
    """Fetch video metadata from a channel within the date range."""

    # Format dates for yt-dlp (YYYYMMDD)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "daterange": yt_dlp.utils.DateRange(start_str, end_str),
    }

    # Handle different input formats:
    # - Full URL: https://www.youtube.com/@Handle or https://www.youtube.com/channel/UCxxx
    # - Handle: @Handle
    # - Channel ID: UCxxx
    if channel_input.startswith("http"):
        # Already a URL - ensure it points to videos tab
        channel_url = channel_input.rstrip("/")
        if not channel_url.endswith("/videos"):
            channel_url += "/videos"
    elif channel_input.startswith("@"):
        # Handle format
        channel_url = f"https://www.youtube.com/{channel_input}/videos"
    else:
        # Assume channel ID
        channel_url = f"https://www.youtube.com/channel/{channel_input}/videos"

    print(f"Fetching videos from: {channel_url}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    if not info or "entries" not in info:
        return []

    videos = []
    for entry in info["entries"]:
        if entry:
            videos.append({
                "id": entry.get("id"),
                "title": entry.get("title"),
                "url": entry.get("url") or f"https://www.youtube.com/watch?v={entry.get('id')}",
            })

    return videos


def download_video(video_id: str, video_url: str, output_dir: Path) -> Path | None:
    """Download a single video. Returns path to downloaded file or None on failure."""

    output_template = str(output_dir / f"{video_id}.%(ext)s")

    ydl_opts = {
        "format": VIDEO_FORMAT,
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Find the downloaded file
        for ext in ["mp4", "mkv", "webm"]:
            path = output_dir / f"{video_id}.{ext}"
            if path.exists():
                return path

        # Check for any file starting with video_id
        for f in output_dir.iterdir():
            if f.stem == video_id:
                return f

        return None

    except Exception as e:
        print(f"  Error downloading {video_id}: {e}")
        return None


def extract_frames(
    video_path: Path,
    video_id: str,
    output_dir: Path,
    fps: float = 1.0,
) -> int:
    """
    Extract frames from video at specified fps.
    Frames are saved as {video_id}_{second}.jpg
    Returns the number of frames extracted.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"  Error: Could not open video {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30  # Fallback

    # Calculate frame interval
    frame_interval = int(video_fps / fps)
    if frame_interval < 1:
        frame_interval = 1

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            second = int(frame_count / video_fps)
            output_path = output_dir / f"{video_id}_{second}.jpg"

            # Save with reasonable JPEG quality (85 is good balance)
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            extracted_count += 1

        frame_count += 1

    cap.release()
    return extracted_count


def process_video(
    video: dict,
    download_dir: Path,
    frames_dir: Path,
    fps: float,
    semaphore: threading.Semaphore,
    video_num: int,
    total_videos: int,
):
    """Download video, extract frames, then clean up. Respects semaphore for disk limit."""

    video_id = video["id"]
    video_url = video["url"]

    # Wait for semaphore (limits videos on disk)
    with semaphore:
        print(f"[{video_num}/{total_videos}] Processing: {video['title'][:50]}...")

        # Download
        print(f"  Downloading {video_id}...")
        video_path = download_video(video_id, video_url, download_dir)

        if video_path is None:
            print(f"  Skipping {video_id} - download failed")
            return

        # Extract frames
        print(f"  Extracting frames at {fps} fps...")
        num_frames = extract_frames(video_path, video_id, frames_dir, fps)
        print(f"  Extracted {num_frames} frames from {video_id}")

        # Clean up video file
        try:
            video_path.unlink()
            print(f"  Deleted video file: {video_path.name}")
        except Exception as e:
            print(f"  Warning: Could not delete {video_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube channel videos and extract frames for classification"
    )
    parser.add_argument(
        "channel",
        help="YouTube channel URL, handle (@name), or ID (UCxxx)"
    )
    parser.add_argument(
        "start_date",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "end_date",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to extract (default: 1.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for frames (default: ./output)"
    )
    parser.add_argument(
        "--max-concurrent-videos",
        type=int,
        default=2,
        help="Max videos to keep on disk at once (default: 2)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)"
    )

    args = parser.parse_args()

    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        print(f"Error parsing date: {e}")
        print("Use format: YYYY-MM-DD")
        return 1

    # Setup directories
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    download_dir = output_dir / "downloads"

    frames_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    # Get video list
    videos = get_channel_videos(args.channel, start_date, end_date)

    if not videos:
        print("No videos found in the specified date range.")
        return 0

    print(f"Found {len(videos)} videos to process")

    # Semaphore to limit videos on disk
    semaphore = threading.Semaphore(args.max_concurrent_videos)

    # Process videos in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for i, video in enumerate(videos, 1):
            future = executor.submit(
                process_video,
                video,
                download_dir,
                frames_dir,
                args.fps,
                semaphore,
                i,
                len(videos),
            )
            futures.append(future)

        # Wait for all to complete
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error processing video: {e}")

    # Cleanup download directory
    try:
        download_dir.rmdir()
    except:
        pass

    print(f"\nDone! Frames saved to: {frames_dir}")
    print(f"Frame naming format: {{video_id}}_{{second}}.jpg")

    return 0


if __name__ == "__main__":
    exit(main())
