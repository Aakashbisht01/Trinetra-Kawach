import React from 'react';

function VideoFeed() {
  // The URL of the video feed from your Flask backend
  const videoFeedUrl = "http://127.0.0.1:5000/video_feed";

  return (
    <div className="video-container">
      <h2>Live Surveillance Feed</h2>
      <img src={videoFeedUrl} alt="Live video feed" />
    </div>
  );
}

export default VideoFeed;
