import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import AlertTitle from '@mui/material/AlertTitle';
import WarningIcon from '@mui/icons-material/Warning';
import LocalFireDepartmentIcon from '@mui/icons-material/LocalFireDepartment';
import SensorsIcon from '@mui/icons-material/Sensors';
import PlayCircleIcon from '@mui/icons-material/PlayCircle';
import Link from '@mui/material/Link';
import { Grid, Typography } from '@mui/material';

const SOCKET_URL = 'https://c89491703810.ngrok-free.app'; // Your backend URL

const App = () => {
  const [openSnackbar, setOpenSnackbar] = useState(false);
  const [currentNotification, setCurrentNotification] = useState(null);
  const audioRef = useRef(null);

  useEffect(() => {
    const socket = io(SOCKET_URL, {
      transports: ['websocket'],
      allowEIO3: true,
    });

    socket.on('connect', () => {
      console.log('✅ Connected to WebSocket server');
    });

    socket.on('disconnect', () => {
      console.log('⚠️ Disconnected from WebSocket server');
    });

    socket.on('notification', (data) => {
      if (data.type === 'crowd') {
        const threshold = 2;
        if (data.person_count >= threshold) {
          const alertData = {
            type: 'crowd',
            timestamp: data.timestamp,
            message: `🚨 Too many people detected! Count: ${data.person_count}`,
          };
          triggerAlert(alertData);
        }
      } else {
        triggerAlert(data);
      }
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  // Unlock audio autoplay with first user interaction
  useEffect(() => {
    const unlockAudio = () => {
      if (audioRef.current) {
        audioRef.current.play().then(() => {
          audioRef.current.pause();
          audioRef.current.currentTime = 0;
          console.log('🔓 Audio unlocked');
        }).catch((err) => {
          console.warn('🔇 Audio unlock failed:', err);
        });
      }
      document.removeEventListener('click', unlockAudio);
    };
    document.addEventListener('click', unlockAudio);
  }, []);

  // Play sound + show snackbar
  const triggerAlert = (alertData) => {
    setCurrentNotification(alertData);
    setOpenSnackbar(true);

    if (audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play().catch((err) => {
        console.warn('🔇 Sound playback failed:', err);
      });
    }
  };

  const handleCloseSnackbar = () => {
    setOpenSnackbar(false);
  };

  const getNotificationIcon = () => {
    const type = currentNotification?.type;
    switch (type) {
      case 'fire':
        return <LocalFireDepartmentIcon fontSize="60px" />;
      case 'hand-gesture':
        return <WarningIcon fontSize="60px" />;
      case 'crowd':
        return <SensorsIcon fontSize="50px" style={{ color: '#FFD700' }} />;
      default:
        return <SensorsIcon fontSize="60px" />;
    }
  };

  const getAlertBackground = () => {
    const type = currentNotification?.type;
    switch (type) {
      case 'fire':
        return '#FF6B00';
      case 'hand-gesture':
        return '#CC3300';
      case 'crowd':
        return '#FFD700';
      default:
        return '#FF0000';
    }
  };

  return (
    <div>
      {/* 🔊 Single audio source */}
      <audio ref={audioRef} src="/alert.mp3" preload="auto" />

      <Snackbar
        open={openSnackbar}
        autoHideDuration={12000}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        sx={{ top: 0, width: '100%', maxWidth: '500px', margin: 'auto' }}
      >
        <Alert
          elevation={6}
          onClose={handleCloseSnackbar}
          variant="filled"
          icon={getNotificationIcon()}
          style={{
            color: '#ffffff',
            backgroundColor: getAlertBackground(),
            width: '100%',
            wordWrap: 'break-word',
            whiteSpace: 'normal',
            lineHeight: 1.4,
            fontSize: '15px',
          }}
        >
          <Grid container spacing={1}>
            <Grid item xs={12}>
              {currentNotification?.location && (
                <AlertTitle style={{ fontSize: '20px' }}>{currentNotification.location}</AlertTitle>
              )}
              {currentNotification?.message && (
                <Typography style={{ fontSize: '15px', whiteSpace: 'pre-line' }}>
                  {currentNotification.message}
                </Typography>
              )}
            </Grid>

            <Grid item xs={8}>
              {currentNotification?.timestamp && (
                <Typography style={{ fontSize: '13px' }}>{currentNotification.timestamp}</Typography>
              )}
            </Grid>

            <Grid item xs={4} style={{ textAlign: 'right' }}>
              {currentNotification?.video_link && (
                <Link href={currentNotification.video_link} target="_blank" rel="noopener noreferrer">
                  <PlayCircleIcon style={{ fontSize: '35px', color: '#ffffff' }} />
                </Link>
              )}
            </Grid>
          </Grid>
        </Alert>
      </Snackbar>
    </div>
  );
};

export default App;
