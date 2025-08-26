import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import './App.css';
import VideoFeed from './components/VideoFeed';
import AlertBanner from './components/AlertBanner';
import StatusPanel from './components/StatusPanel';
import EventLog from './components/EventLog';

// Connect to your Python backend server
const SOCKET_URL = "http://127.0.0.1:5000";
const socket = io(SOCKET_URL);

function App() {
  const [status, setStatus] = useState({
    video: { label: "Connecting...", probability: 0.0 },
    audio: { label: "Connecting...", probability: 0.0 },
    fused_alert: "CONNECTING"
  });
  const [eventLog, setEventLog] = useState([]);

  useEffect(() => {
    // Listen for 'update_status' messages from the backend
    socket.on('update_status', (data) => {
      // Check if a new alert needs to be logged
      if (data.fused_alert !== "NORMAL" && status.fused_alert === "NORMAL") {
        const newEvent = {
          time: new Date().toLocaleTimeString(),
          level: data.fused_alert,
          video: data.video.label,
          audio: data.audio.label
        };
        setEventLog(prevLog => [newEvent, ...prevLog]);
      }
      setStatus(data);
    });

    // Clean up the connection when the component unmounts
    return () => {
      socket.off('update_status');
    };
  }, [status]); // Dependency array includes status to access the previous state

  return (
    <div className="App">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2L3 5V11C3 16.5 6.8 21.7 12 23C17.2 21.7 21 16.5 21 11V5L12 2Z" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="12" cy="10" r="1" fill="white"/>
              <circle cx="9" cy="13" r="1" fill="white"/>
              <circle cx="15" cy="13" r="1" fill="white"/>
            </svg>
          </div>
          <h1>Trinetra Kavach</h1>
        </div>
        <StatusPanel status={status} />
        <EventLog events={eventLog} />
      </aside>
      <main className="main-content">
        <AlertBanner status={status.fused_alert} />
        <VideoFeed />
      </main>
    </div>
  );
}

export default App;
