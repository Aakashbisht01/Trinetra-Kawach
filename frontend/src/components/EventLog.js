import React from 'react';

function EventLog({ events }) {
  return (
    <div className="event-log">
      <h2>Event Log</h2>
      <div className="log-container">
        {events.length === 0 ? (
          <p className="no-events">No events recorded yet.</p>
        ) : (
          <ul>
            {events.map((event, index) => (
              <li key={index} className={log-item ${event.level.toLowerCase()}}>
                <strong>{event.time}</strong> - {event.level.replace('_', ' ')}
                <small> (Video: {event.video} | Audio: {event.audio})</small>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

export default EventLog;
