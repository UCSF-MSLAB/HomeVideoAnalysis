import sys
from pytube import YouTube
from dotenv import dotenv_values

secrets = dotenv_values(".env")

# reference
# https://www.geeksforgeeks.org/download-youtube-videos-or-whole-playlist-with-python/

# example Youtube playlist id: PLCF7983F60455753E


def download_youtube_video(video_id, download_path):

    link = f'https://www.youtube.com/watch?v={video_id}'
    yt_obj = YouTube(link)
    filters = yt_obj.streams.filter(progressive=True,
                                    file_extension='mp4')
    filters.get_highest_resolution().download(download_path)

    pass


def main():

    args = sys.argv
    download_youtube_video(args[1], args[2])


if __name__ == '__main__':
    main()
