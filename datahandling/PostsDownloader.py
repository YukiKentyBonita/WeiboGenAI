from weibo_crawler import Profile, Follow, Weibos
from pathlib import Path
import os

cookies = os.getenv("WEIBO_COOKIES", "")

def _raw_path(filename: str) -> Path:
    base_dir = Path(__file__).resolve().parent
    outputfile = base_dir.parent / "data" / "raw" / filename
    outputfile.parent.mkdir(parents=True, exist_ok=True)
    return outputfile

def get_user_profile(username: str):
    outputfile = _raw_path("userprofile.csv")
    userprofile = Profile(csvfile=outputfile, delay=1, cookies=cookies)
    try:
        result = userprofile.get_profile(userid=username)
        print("=== Raw profile result from crawler ===")
        print(result)
        print("=== CSV saved to:", outputfile, "===")
        return result
    except Exception as e:
        print("⚠️ Error crawling profile:", e)
        return None

def get_user_follows(username: str):
    outputfile = _raw_path("follows.csv")
    follow = Follow(csvfile=outputfile, delay=1, cookies=cookies)
    try:
        follow.follow_who(userid=username)
        print("✅ Finished crawling follows. Saved to:", outputfile)
        return True
    except AttributeError as e:
        print("⚠️ Skipping follow list due to crawler error:", e)
        return False
    except Exception as e:
        print("⚠️ Unexpected error when crawling follows:", e)
        return False

def get_user_posts(username: str):
    outputfile = _raw_path("posts.csv")
    posts = Weibos(csvfile=outputfile, delay=1, cookies=cookies)
    try:
        posts.get_weibos_by_userid(userid=username)
        print("✅ Finished crawling posts. Saved to:", outputfile)
        return True
    except Exception as e:
        print("⚠️ Error crawling posts:", e)
        return False

if __name__ == "__main__":
    username = ""  # example username
    print(f"Starting to crawl data for user: {username}")
    get_user_profile(username)
    get_user_follows(username)
    get_user_posts(username)
    print("Crawling completed.")