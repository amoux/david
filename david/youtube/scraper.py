import json
import time

import lxml.html
import requests
from lxml.cssselect import CSSSelector


class YTCommentScraper(object):
    USER_AGENT = "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 \
        (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36"

    def __init__(self, retries=10, sleep_per_retry=20, order_by_time=True):
        self.order_by_time = order_by_time
        self.retries = retries
        self.sleep_per_retry = sleep_per_retry

    def _from_html_keys(self, html, key, num_chars=2):
        pos_begin = html.find(key) + len(key) + num_chars
        pos_end = html.find('"', pos_begin)
        return html[pos_begin: pos_end]

    def _extract_comment_content(self, html):
        tree = lxml.html.fromstring(html)
        item_sel = CSSSelector('.comment-item')
        text_sel = CSSSelector('.comment-text-content')
        time_sel = CSSSelector('.time')
        author_sel = CSSSelector('.user-name')
        for item in item_sel(tree):
            yield {'cid': item.get('data-cid'),
                   'text': text_sel(item)[0].text_content(),
                   'time': time_sel(item)[0].text_content().strip(),
                   'author': author_sel(item)[0].text_content()}

    def _extract_from_comment_replies(self, html):
        tree = lxml.html.fromstring(html)
        select = CSSSelector('.comment-replies-header > .load-comments')
        return [i.get('data-cid') for i in select(tree)]

    def _ajax_request(self, session, url, params, data):
        for _ in range(self.retries):
            response = session.post(url, params=params, data=data)
            if response.status_code == 200:
                response_dict = json.loads(response.text)
                return response_dict.get(
                    'page_token', None), response_dict['html_content']
            else:
                time.sleep(self.sleep_per_retry)

    def scrape_comments(self, video_id: str, sleep=1):
        """Scrapes Comments from a youtube video id."""
        session = requests.Session()
        session.headers['User-Agent'] = self.USER_AGENT
        response = session.get(
            "https://www.youtube.com/all_comments?v={}".format(video_id))
        html = response.text
        reply_cids = self._extract_from_comment_replies(html)
        ret_cids = list()
        for comment in self._extract_comment_content(html):
            ret_cids.append(comment['cid'])
            yield comment

        page_token = self._from_html_keys(html, 'data-token')
        session_token = self._from_html_keys(html, 'XSRF_TOKEN', 4)
        first_iteration = True
        while page_token:
            data = {'video_id': video_id, 'session_token': session_token}
            params = {
                'action_load_comments': 1,
                'order_by_time': self.order_by_time,
                'filter': video_id
            }
            if first_iteration:
                params['order_menu'] = True
            else:
                data['page_token'] = page_token
            response = self._ajax_request(
                session, "https://www.youtube.com/comment_ajax", params, data)
            if not response:
                break

            page_token, html = response
            reply_cids += self._extract_from_comment_replies(html)
            for page in self._extract_comment_content(html):
                if page['cid'] not in ret_cids:
                    ret_cids.append(page['cid'])
                    yield page

            first_iteration = False
            time.sleep(sleep)

        for cid in reply_cids:
            data = {'comment_id': cid, 'video_id': video_id,
                    'can_reply': 1, 'session_token': session_token}
            params = {'action_load_replies': 1, 'order_by_time': True,
                      'filter': video_id, 'tab': 'inbox'}

            response = self._ajax_request(
                session, self.YOUTUBE_AJAX_URL, params, data)
            if not response:
                break

            _, html = response
            for page in self._extract_comment_content(html):
                if page['cid'] not in ret_cids:
                    ret_cids.append(page['cid'])
                    yield page
            time.sleep(sleep)