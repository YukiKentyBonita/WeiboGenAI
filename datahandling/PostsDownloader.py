from weibo_crawler import Profile, Follow, Weibos, Comments
import os

cookies = '_T_WM=471e6dbd07fa39381b461368c4827c2e; SCF=Aleq5KUilV79SaEajH35shx0Kk77f3CftY-jWSbEAySempYkwPyUi1IHwEnCqjftqe4P4tpOP3_NiZS-kfGYRjw.; SUB=_2A25F_Fh_DeRhGeRP7VcV-C3JzjmIHXVncNW3rDV6PUJbktCOLU3QkW1NUAZ-TWjDNqI05AV2-2FpvyhnsPCO9Jx6; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W5xsiCWuHlhulE_DVnFZ6Np5NHD95QEeKqfShn0SK-fWs4DqcjZdJLfdJLfxPeEMBtt; SSOLoginState=1761093679; ALF=1763685679; MLOGIN=1; M_WEIBOCN_PARAMS=lfid%3D1076031860563805%26luicode%3D20000174'

def get_user_profile(username):
    outputfile = os.getcwd() + "/userprofile.csv"
    userprofile = Profile(
        csvfile=outputfile,
        delay=1,
        cookies=cookies
    )

    # Call the crawler and CAPTURE the return value
    result = userprofile.get_profile(userid=username)

    print("=== Raw profile result from crawler ===")
    print(result)
    print("=== CSV saved to:", outputfile, "===")

# def get_user_follows(username):
#     outputfile = os.getcwd() + "/follows.csv"
#     follow = Follow(
#         csvfile=outputfile,
#         delay=1,
#         cookies=cookies
#     ) 
#     follow.follow_who(userid=username)
#     print(follow)

def get_user_follows(username):
    outputfile = os.getcwd() + "/follows.csv"
    follow = Follow(
        csvfile=outputfile,
        delay=1,
        cookies=cookies
    )
    try:
        follow.follow_who(userid=username)
        print("✅ Finished crawling follows. Saved to:", outputfile)
        print(follow)
    except AttributeError as e:
        # Crawler couldn't find the max page number
        print("⚠️ Skipping follow list due to crawler error:", e)
        print("   This usually means the library couldn't detect the max page number")
        print("   (likely caused by a change in Weibo's page layout).")
    except Exception as e:
        # Something else goes wrong
        print("⚠️ Unexpected error when crawling follows, skipping this step:", e)


def get_user_posts(username):
    outputfile = os.getcwd() + "/posts.csv"
    posts = Weibos(
        csvfile=outputfile,
        delay=1,
        cookies=cookies
    ) 
    posts.get_weibos_by_userid(userid=username)

if __name__ == "__main__":
    get_user_profile('1860563805')
    # get_user_follows('1860563805')
    # get_user_posts('1860563805')
    print("Data download completed.")