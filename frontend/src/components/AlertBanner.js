import React from 'react';

function AlertBanner({ status }) {
  const getBannerClass = () => {
    switch (status) {
      case 'HIGH_ALERT':
        return 'banner-high';
      case 'LOW_ALERT':
        return 'banner-low';
      default:
        return 'banner-normal';
    }
  };

  const getBannerText = () => {
    switch (status) {
      case 'HIGH_ALERT':
        return ' HIGH ALERT: Distress Detected! ';
      case 'LOW_ALERT':
        return ' LOW ALERT: Potential Anomaly Detected';
      case 'CONNECTING':
        return ' Connecting to server...';
      default:
        return ' STATUS: NORMAL';
    }
  };

  return (
    <div className={alert-banner ${getBannerClass()}}>
      {getBannerText()}
    </div>
  );
}

export default AlertBanner;
