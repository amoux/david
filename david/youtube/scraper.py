import datetime
import json
import logging
import sys
import time

import lxml.html
import requests
from dateutil.relativedelta import relativedelta
from lxml.cssselect import CSSSelector
from tqdm import tqdm

from ..text import normalize_whitespace, unicode_to_ascii
from .utils import extract_videoid

logger = logging.getLogger(__name__)


def to_timestamp(timestamp: str, time_format="%m/%d/%Y, %H:%M:%S"):
    """Convert YouTube comments timestamps to datetime.

    Note: If the date type is 'days', 'months', or 'years' the time used is not
        The actual time of the posted comment as the only information for these
        types lacks information from types with the format:
        `'1 hour ago' or '3 hours ago'`.

    Usage:
        >>> timestamp_to_datetime('3 days ago', "%m/%d/%Y, %H:%M:%S")
        >>> '02/24/2020, 22:01:22'

    """
    if not isinstance(timestamp, str):
        timestamp = str(timestamp).lower()

    now = datetime.datetime.now()
    stamp = timestamp.split()

    if stamp[1].startswith("d"):
        date = now - datetime.timedelta(days=int(stamp[0]))
        return date.strftime(time_format)
    elif stamp[1].startswith("h"):
        date = now - datetime.timedelta(hours=int(stamp[0]))
        return date.strftime(time_format)
    elif stamp[1].startswith("mi"):
        date = now - datetime.timedelta(minutes=int(stamp[0]))
        return date.strftime(time_format)
    elif stamp[1].startswith("w"):
        date = now - relativedelta(weeks=int(stamp[0]))
        return date.strftime(time_format)
    elif stamp[1].startswith("mo"):
        date = now - relativedelta(months=int(stamp[0]))
        return date.strftime(time_format)
    elif stamp[2].startswith("y"):
        date = now - relativedelta(years=int(stamp[0]))
        return date.strftime(time_format)


class YTCommentScraper(object):
    """YouTube Comments Scraper Class."""

    YOUTUBE_AJAX_URL = "https://www.youtube.com/comment_ajax"
    USER_AGENT = "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 \
        (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36"

    def __init__(
        self,
        retries=10,
        sleep_per_retry=20,
        order_by_time=True,
        format_time=True,
        time_format="%m/%d/%Y, %H:%M:%S",
    ):
        self.order_by_time = order_by_time
        self.retries = retries
        self.sleep_per_retry = sleep_per_retry
        self.format_time = format_time
        self.time_format = time_format

    def _from_html_keys(self, html, key, num_chars=2):
        pos_begin = html.find(key) + len(key) + num_chars
        pos_end = html.find('"', pos_begin)
        return html[pos_begin:pos_end]

    def _extract_comment_content(self, html):
        tree = lxml.html.fromstring(html)
        item_sel = CSSSelector(".comment-item")
        text_sel = CSSSelector(".comment-text-content")
        time_sel = CSSSelector(".time")
        author_sel = CSSSelector(".user-name")

        for item in item_sel(tree):
            text = normalize_whitespace(
                unicode_to_ascii(text_sel(item)[0].text_content()))
            post_time = time_sel(item)[0].text_content().strip()
            if self.format_time:
                post_time = to_timestamp(post_time, self.time_format)

            yield {
                "cid": item.get("data-cid"),
                "text": text,
                "time": post_time,
                "author": author_sel(item)[0].text_content(),
            }

    def _extract_from_comment_replies(self, html):
        tree = lxml.html.fromstring(html)
        select = CSSSelector(".comment-replies-header > .load-comments")
        return [i.get("data-cid") for i in select(tree)]

    def _ajax_request(self, session, url, params, data):
        for _ in range(self.retries):
            response = session.post(url, params=params, data=data)
            if response.status_code == 200:
                response_dict = json.loads(response.text)
                return (
                    response_dict.get("page_token", None),
                    response_dict["html_content"],
                )
            else:
                time.sleep(self.sleep_per_retry)

    def _yield_scraped_items(self, video_id=None, video_url=None, sleep=1):
        if video_url:
            video_id = extract_videoid(video_url)
        if not video_id:
            raise ValueError(f"You need to pass a valid id, not: {video_id}")

        session = requests.Session()
        session.headers["User-Agent"] = self.USER_AGENT
        youtube_video_url = f"https://www.youtube.com/all_comments?v={video_id}"
        response = session.get(youtube_video_url)

        html = response.text
        reply_cids = self._extract_from_comment_replies(html)

        ret_cids = list()
        for comment in self._extract_comment_content(html):
            ret_cids.append(comment["cid"])
            yield comment

        page_token = self._from_html_keys(html, "data-token")
        session_token = self._from_html_keys(html, "XSRF_TOKEN", 4)
        first_iteration = True

        while page_token:
            data = {"video_id": video_id, "session_token": session_token}
            params = {
                "action_load_comments": 1,
                "order_by_time": self.order_by_time,
                "filter": video_id,
            }
            if first_iteration:
                params["order_menu"] = True
            else:
                data["page_token"] = page_token

            response = self._ajax_request(session, self.YOUTUBE_AJAX_URL, params, data)
            if not response:
                logger.error("No session response %s", session)
                break

            page_token, html = response
            reply_cids += self._extract_from_comment_replies(html)
            for page in self._extract_comment_content(html):
                if page["cid"] not in ret_cids:
                    ret_cids.append(page["cid"])
                    yield page

            first_iteration = False
            time.sleep(sleep)

        for cid in reply_cids:
            data = {
                "comment_id": cid,
                "video_id": video_id,
                "can_reply": 1,
                "session_token": session_token,
            }
            params = {
                "action_load_replies": 1,
                "order_by_time": True,
                "filter": video_id,
                "tab": "inbox",
            }
            response = self._ajax_request(session, self.YOUTUBE_AJAX_URL, params, data)
            if not response:
                logger.error("No session response %s", session)
                break

            _, html = response
            for page in self._extract_comment_content(html):
                if page["cid"] not in ret_cids:
                    ret_cids.append(page["cid"])
                    yield page

            time.sleep(sleep)

    def scrape_comments_generator(
        self, video_id: str = None, video_url: str = None, limit: int = None, sleep=1
    ):
        """Scrapes Comments from a video id or a video url.

        `video_id`: The video id to use for extracting the comments:
        `video_url`: The the url containing the video id. It uses a method which can
            handle multiple forms of an url or string with the id. For more
            info check: `david.youtube.utils.extract_videoid`.
        `limit`: Limit the number of comments to download. The scraper will scrape all
            available comments if limit=None.
        """
        count = 0
        for comment in self._yield_scraped_items(video_id, video_url, sleep):
            yield comment
            count += 1
            sys.stdout.write(f"* 🛰 {count} comments scraped\r")
            sys.stdout.flush()
            if limit and count >= limit:
                break

    def scrape_comments(
        self, video_id: str = None, video_url: str = None, limit: int = None, sleep=1
    ):
        """Scrapes Comments from a video id or a video url.

        `video_id`: The video id to use for extracting the comments:
        `video_url`: The the url containing the video id. It uses a method which can
            handle multiple forms of an url or string with the id. For more
            info check: `david.youtube.utils.extract_videoid`.
        `limit`: Limit the number of comments to download. The scraper will scrape all
            available comments if limit=None.
        """
        return list(self.scrape_comments_generator(video_id, video_url, limit, sleep))
