import praw
import json

client_id = "tkFn8F1DLLMsHxP9R8f_wA"
client_secret = "DVd2kB_bnjO-knDnEbbpjng5iZYnkg"
user_agent = "app by u/Busy-Welcome-4696"

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

subreddit = reddit.subreddit("politics")

posts_data = []
for post in subreddit.search("money", limit=100):
    posts_data.append({
        "title": post.title,
        "score": post.score,
        "url": post.url,
        "created_utc": post.created_utc,
        "num_comments": post.num_comments,
        "selftext": post.selftext
    })

with open("reddit_posts.json", "w", encoding="utf-8") as f:
    json.dump(posts_data, f, ensure_ascii=False, indent=4)
