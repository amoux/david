from __future__ import print_function

import io
import json
import os
import sys
import time
from os.path import exists, isfile, join

import lxml.html
import requests as _requests
from lxml.cssselect import CSSSelector as _CSSSelector

_YT_COMMENTS_URL = 'https://www.youtube.com/all_comments?v={youtube_id}'
_YT_COMMENTS_AJAX_URL = 'https://www.youtube.com/comment_ajax'
_USER_AGENT = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 \
    (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36'


def _from_html_keys(html, key, num_chars=2):
    pos_begin = html.find(key) + len(key) + num_chars
    pos_end = html.find('"', pos_begin)
    return html[pos_begin: pos_end]


def _extract_comment_content(html):
    tree = lxml.html.fromstring(html)
    item_sel = _CSSSelector('.comment-item')
    text_sel = _CSSSelector('.comment-text-content')
    time_sel = _CSSSelector('.time')
    author_sel = _CSSSelector('.user-name')

    for item in item_sel(tree):
        yield {'cid': item.get('data-cid'),
               'text': text_sel(item)[0].text_content(),
               'time': time_sel(item)[0].text_content().strip(),
               'author': author_sel(item)[0].text_content()}


def _extract_from_comment_replies(html):
    tree = lxml.html.fromstring(html)
    select = _CSSSelector('.comment-replies-header > .load-comments')
    return [i.get('data-cid') for i in select(tree)]


def _ajax_request(session, url, params, data, retries=10, sleep=20):
    '''Sends a POST request. Returns Response object.
    Parameters:
    `url` : URL for the new Request object.
    `data` : (optional)
        Dictionary, bytes, or file-like object to send in the
        body of the Request.
    `json` : (optional) json to send in the body of the Request.
    `**kwargs` :  Optional arguments that request takes.
    '''
    for _ in range(retries):
        response = session.post(url, params=params, data=data)
        if response.status_code == 200:
            response_dict = json.loads(response.text)
            return response_dict.get(
                'page_token', None), response_dict['html_content']
        else:
            time.sleep(sleep)


def _scrape_comments(youtube_id, sleep=1):
    session = _requests.Session()
    session.headers['User-Agent'] = _USER_AGENT
    response = session.get(_YT_COMMENTS_URL.format(youtube_id=youtube_id))
    html = response.text
    reply_cids = _extract_from_comment_replies(html)

    ret_cids = []
    for comment in _extract_comment_content(html):
        ret_cids.append(comment['cid'])
        yield comment

    page_token = _from_html_keys(html, 'data-token')
    session_token = _from_html_keys(html, 'XSRF_TOKEN', 4)

    first_iteration = True
    while page_token:
        data = {
            'video_id': youtube_id,
            'session_token': session_token
        }
        params = {
            'action_load_comments': 1,
            'order_by_time': True,
            'filter': youtube_id
        }
        if first_iteration:
            params['order_menu'] = True
        else:
            data['page_token'] = page_token

        response = _ajax_request(session, _YT_COMMENTS_AJAX_URL, params, data)
        if not response:
            break

        page_token, html = response
        reply_cids += _extract_from_comment_replies(html)

        for page in _extract_comment_content(html):
            if page['cid'] not in ret_cids:
                ret_cids.append(page['cid'])
                yield page

        first_iteration = False
        time.sleep(sleep)

    for cid in reply_cids:
        data = {'comment_id': cid,
                'video_id': youtube_id,
                'can_reply': 1,
                'session_token': session_token}

        params = {'action_load_replies': 1,
                  'order_by_time': True,
                  'filter': youtube_id,
                  'tab': 'inbox'}

        response = _ajax_request(
            session, _YT_COMMENTS_AJAX_URL, params, data)
        if not response:
            break

        _, html = response
        for page in _extract_comment_content(html):
            if page['cid'] not in ret_cids:
                ret_cids.append(page['cid'])
                yield page
        time.sleep(sleep)


def _write2json(fn: str, video_id: str, limit=None):
    '''Jsonline file writer.
    '''
    count = 0
    with io.open(fn, 'w', encoding='utf-8') as fp:
        for comment in _scrape_comments(video_id):
            print(json.dumps(comment, ensure_ascii=False), file=fp)
            count += 1

            sys.stdout.write('mining %d comment(s)\r' % count)
            sys.stdout.flush()

            if limit and count >= limit:
                break
    print(f'done extracting comments')


def download(video_id: str, dirpath='downloads',
             limit=None, load_corpus=False, force_download=False):
    '''
    Downloads comments from a youtube videos.

    NOTE: The files are named after the video id e.g. 4Dk3jOSbz_0.json

    Parameters
    ----------

    `video_id` : (str)
        The youtube id from a video url. For example, '4Dk3jOSbz_0'

    `dirpath` : (str, default='downloads')

    NOTE: I NEED TO ADD A DEFAULT GLOBAL ENVIROMENT VARIABLE TO THE
    LOCATION WHERE THE DOWNLOADS WILL BE SAVED. LIKE DAVID_DATA. I
    ALREADY STARTED WORKING ON THIS WILL SOME METHODS FROM THE SKLEARN
    LIBRARY. LOCATED IN THE MODELS DIRECTORY OF THIS PROJECT.


        The directory where the downloads will be saved.

    `limit` : (int, default=None)
        Sets a limit to the number of comments to download
    '''
    # make directory if it doesnt exist.
    if not exists(dirpath):
        os.makedirs(dirpath)
    fp = join(dirpath, f'{video_id}.json')

    # check if the video has already beed downloaded.
    if (isfile(fp) and not force_download):
        raise Exception(f'The video id: {video_id} already exists!\n'
                        'Set force_download=True to override the exception.')
    _write2json(fp, video_id, limit)

    # if true, return the full path of the scraped file.
    if load_corpus:
        return fp
