import os
import sqlite3
from datetime import datetime

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_sqlalchemy.sqlalchemy import (Column, DateTime, ForeignKey, Integer,
                                         String, Text, relationship)

conn = sqlite3.connect("vuepoint_test.db")
conn.close()


app = Flask('__name__')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////vuepoint_test.db'
db = SQLAlchemy(app)


class User(db.Model):
    id = Column(Integer, primary_key=True)
    user_name = Column(String(20), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    image_file = Column(String(20), nullable=False, default="default.jpg")
    password = Column(String(60), nullable=False)
    posts = relationship('Post', backref='author', lazy=True)
    videos = relationship('Video', backref='author', lazy=True)

    def __repr__(self):
        return (f'User(\n'
                f'  user_name={self.user_name},\n'
                f'  email={self.email},\n'
                f'  posts={self.posts},\n'
                f'  videos={self.videos})')


class Post(db.Model):
    id = Column(Integer, primary_key=True)
    title = Column(String(100), nullable=False)
    text = Column(Text, nullable=False)
    post_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"<Post('{self.title}', '{self.text}', '{self.post_date}')>"


class Video(db.Model):
    id = Column(Integer, primary_key=True)
    video_id = Column(String(100), unique=True, nullable=False)
    search_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    video_info = relationship('VideoSnippet', backref='snippet', lazy=True)
    user_id = Column(Integer, ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return (f'Video(\n'
                f'  video_id={self.video_id},\n'
                f'  search_date={self.search_date},\n'
                f'  user_id={self.user_id})')


class VideoSnippet(db.Model):
    '''
    publishedAt : '2019-10-01T15:41:04.000Z'
    channelId : 'UCbfYPyITQ-7l4upoX8nvctg'
    title : text
    description : text
    thumbnails : video_snippet.thumbnails['default']['url']
    tags : tag2labels(tags)
    categoryId : 28
    defaultAudioLanguage : 'en'
     '''
    id = Column(Integer, primary_key=True)
    published_at = Column(String(20), nullable=False)
    channel_id = Column(String(100), nullable=False)
    title = Column(String(100), nullable=False)
    description = Column(Text)
    thumbnail = Column(String(120), nullable=False)
    tags = Column(Text)
    category_id = Column(Integer, nullable=False)
    language = Column(String(10), nullable=True)
    video_id = Column(Integer, ForeignKey('video.id'), nullable=False)

    def __repr__(self):
        return ('VideoSnippet(\n'
                f'  video_id={self.video_id},\n'
                f'  title={self.title},\n'
                f'  language={self.language},\n'
                f'  tags=[{self.tags}],\n'
                f'  channel_id={self.channel_id})')
