import React, { useState } from "react";
import Bg2 from "../../images/LoginBg2.png";
import {
    Grid,
    TextField,
    Button,
} from "@mui/material";
import GppGoodIcon from "@mui/icons-material/GppGood";
import { useNavigate } from "react-router-dom";
import { auth } from "../../Firebase";
import { signInWithEmailAndPassword } from "firebase/auth";

const Login = () => {
    const navigate = useNavigate();
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");

    const onLogin = (e) => {
        e.preventDefault();
        signInWithEmailAndPassword(auth, email, password)
            .then((userCredential) => {
                const user = userCredential.user;
                navigate("/dashboard/managers");
                console.log(user);
            })
            .catch((error) => {
                console.log(error.code, error.message);
            });
    };

    return (
        <div
            style={{
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
                alignItems: "center",
                backgroundImage: `url(${Bg2})`,
                backgroundSize: "cover",
                backgroundPosition: "center",
                width: "100%",
                height: "100vh",
            }}
        >
            <Grid container>
                <Grid item xs={7}>
                    <h1
                        style={{
                            color: "white",
                            fontSize: "8rem",
                        }}
                    >
                        SafetyNet.
                    </h1>
                </Grid>
                <Grid item xs={5}>
                    <div
                        style={{
                            backgroundColor: "white",
                            borderRadius: "30px",
                            margin: "2rem",
                            padding: "1rem",
                            display: "flex",
                            flexDirection: "column",
                            justifyContent: "center",
                            alignItems: "center",
                        }}
                    >
                        <GppGoodIcon sx={{ fontSize: 60, color: "#0530AD" }} />
                        <h1 style={{ marginTop: "0" }}>SafetyNet.</h1>
                        <h2 style={{ marginTop: "0" }}>Manager Login</h2>

                        {/* Toggle removed - only manager supported */}
                        <TextField
                            label="Username"
                            variant="outlined"
                            onChange={(e) => setEmail(e.target.value)}
                            style={{ width: "80%" }}
                        />
                        <TextField
                            label="Password"
                            type="password"
                            variant="outlined"
                            onChange={(e) => setPassword(e.target.value)}
                            style={{ marginTop: "1rem", width: "80%" }}
                        />
                        <Button
                            variant="contained"
                            onClick={onLogin}
                            disabled={!email || !password}
                            style={{
                                backgroundColor: "#0530AD",
                                borderRadius: "30px",
                                padding: "1rem 2rem",
                                marginTop: "1rem",
                            }}
                        >
                            Login
                        </Button>

                    </div>
                </Grid>
            </Grid>
        </div>
    );
};

export default Login;
