import React  from "react";
import Navbar from "../Navbar";
import Banner from "./Banner";
import Goals from "./Aboutus";
import Tech from "./Technology";
import PreFoot from "./PreFoot";
import Footer from "./footer";
const HomePage = () => {
    return (
        <div>
            <Navbar />
            <Banner />
            <Goals />
            <Tech />
            <PreFoot />
            <Footer />
        </div>
    )
}

export default HomePage;