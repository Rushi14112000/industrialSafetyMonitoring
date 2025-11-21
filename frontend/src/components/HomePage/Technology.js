// import React from 'react';
// import { Grid, Typography } from '@mui/material';
// import GppGoodIcon from '@mui/icons-material/GppGood';
// import WavingHandIcon from '@mui/icons-material/WavingHand';
// import ReportProblemIcon from '@mui/icons-material/ReportProblem';
// import SensorsIcon from '@mui/icons-material/Sensors';

// const Tech = () => {
//     return (
//         <div style={{ padding: '2rem', textAlign: 'center' }}>
//             <Typography 
//                 variant="h2" 
//                 style={{
//                     marginTop: '8rem',
//                     fontWeight: 'bold',
//                 }}
//             >
//                 Our technologies that benefit you
//             </Typography>

//             <Grid 
//                 container 
//                 spacing={3} 
//                 justifyContent="center"   // ⬅ centers Grid items
//                 style={{ marginTop: '2rem' }}
//             >

//                 <Grid item xs={12} sm={6} md={3}>   {/* Responsive */}
//                     <div style={{ padding: '1rem' }}>
//                         <GppGoodIcon color='primary' sx={{ fontSize: 60 }} />
//                         <h2 style={{ fontWeight: 'bold' }}>Safety Gear Detection</h2>
//                         <p>Workers could be monitored to check if they are wearing safety gear like helmets, gloves, etc.</p>
//                     </div>
//                 </Grid>

//                 <Grid item xs={12} sm={6} md={3}>
//                     <div style={{ padding: '1rem' }}>
//                         <WavingHandIcon color='primary' sx={{ fontSize: 60 }} />
//                         <h2 style={{ fontWeight: 'bold' }}>Hand Gesture Recognition</h2>
//                         <p>Workers could use hand gestures to control machines or alert authorities in an emergency.</p>
//                     </div>
//                 </Grid>

//                 <Grid item xs={12} sm={6} md={3}>
//                     <div style={{ padding: '1rem' }}>
//                         <ReportProblemIcon color='primary' sx={{ fontSize: 60 }} />
//                         <h2 style={{ fontWeight: 'bold' }}>Fire Detection</h2>
//                         <p>Factory fires could be detected early and prevented using real-time alerts.</p>
//                     </div>
//                 </Grid>

//             </Grid>
//         </div>
//     );
// };

// export default Tech;



import React from 'react';
import { Grid, Typography } from '@mui/material';
import GppGoodIcon from '@mui/icons-material/GppGood';
import WavingHandIcon from '@mui/icons-material/WavingHand';
import ReportProblemIcon from '@mui/icons-material/ReportProblem';

const Tech = () => {
    return (
        <div id="services" style={{ padding: '2rem', textAlign: 'center' }}>
            <Typography 
                variant="h2" 
                style={{
                    marginTop: '8rem',
                    fontWeight: 'bold',
                }}
            >
                Our technologies that benefit you
            </Typography>

            <Grid 
                container 
                spacing={3} 
                justifyContent="center"
                style={{ marginTop: '2rem' }}
            >

                <Grid item xs={12} sm={6} md={3}>
                    <div style={{ padding: '1rem' }}>
                        <GppGoodIcon color='primary' sx={{ fontSize: 60 }} />
                        <h2 style={{ fontWeight: 'bold' }}>Safety Gear Detection</h2>
                        <p>Workers could be monitored to check if they are wearing safety gear like helmets, gloves, etc.</p>
                    </div>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                    <div style={{ padding: '1rem' }}>
                        <WavingHandIcon color='primary' sx={{ fontSize: 60 }} />
                        <h2 style={{ fontWeight: 'bold' }}>Hand Gesture Recognition</h2>
                        <p>Workers could use hand gestures to control machines or alert authorities in an emergency.</p>
                    </div>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                    <div style={{ padding: '1rem' }}>
                        <ReportProblemIcon color='primary' sx={{ fontSize: 60 }} />
                        <h2 style={{ fontWeight: 'bold' }}>Fire Detection</h2>
                        <p>Factory fires could be detected early and prevented using real-time alerts.</p>
                    </div>
                </Grid>

            </Grid>
        </div>
    );
};

export default Tech;
