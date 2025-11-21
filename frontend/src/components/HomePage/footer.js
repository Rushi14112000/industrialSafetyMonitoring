import { Box, Grid, Typography, Link } from "@mui/material";
import FacebookIcon from "@mui/icons-material/Facebook";
import InstagramIcon from "@mui/icons-material/Instagram";
import LinkedInIcon from "@mui/icons-material/LinkedIn";

const Footer = () => {
  return (
    <Box
      sx={{
        backgroundColor: "#1a1a1a",
        color: "#fff",
        padding: "3rem 2rem",
        marginTop: "4rem",
      }}
    >
      <Grid
        container
        spacing={4}
        justifyContent="center"
        alignItems="flex-start"     // 🔹 ensures top alignment
        textAlign="left"           // 🔹 makes column text aligned
      >

        {/* Company Section */}
        <Grid item xs={12} sm={6} md={3}>
          <Box sx={{ maxWidth: "250px", margin: "auto" }}>   {/* Center column */}
            <Typography variant="h6" fontWeight="bold" gutterBottom>
              Industrial Safety Monitoring
            </Typography>
            <Typography variant="body2">
              AI-based safety solutions for industries to ensure real-time
              monitoring, hazard detection, and worker protection.
            </Typography>
          </Box>
        </Grid>

        {/* Contact Info */}
        <Grid item xs={12} sm={6} md={3}>
          <Box sx={{ maxWidth: "250px", margin: "auto" }}>   {/* Center column */}
            <Typography variant="h6" fontWeight="bold" gutterBottom>
              Contact Us
            </Typography>
            <Typography variant="body2">📍 Pune, Maharashtra, India</Typography>
            <Typography variant="body2">📧 support@safetymonitor.com</Typography>
            <Typography variant="body2">📞 +91 9876543210</Typography>
          </Box>
        </Grid>

        {/* Social Media */}
        <Grid item xs={12} sm={6} md={3}>
          <Box sx={{ maxWidth: "250px", margin: "auto" }}>   {/* Center column */}
            <Typography variant="h6" fontWeight="bold" gutterBottom>
              Follow Us
            </Typography>

            <Box sx={{ display: "flex", gap: "1rem" }}>
              <Link href="#" color="inherit"><FacebookIcon fontSize="large" /></Link>
              <Link href="#" color="inherit"><InstagramIcon fontSize="large" /></Link>
              <Link href="#" color="inherit"><LinkedInIcon fontSize="large" /></Link>
            </Box>
          </Box>
        </Grid>
      </Grid>

      {/* Bottom Bar */}
      <Box
        sx={{
          textAlign: "center",
          marginTop: "3rem",
          borderTop: "1px solid #444",
          paddingTop: "1rem",
        }}
      >
        <Typography variant="body2">
          © {new Date().getFullYear()} Industrial Safety Monitoring — All rights reserved.
        </Typography>
      </Box>
    </Box>
  );
};

export default Footer;
