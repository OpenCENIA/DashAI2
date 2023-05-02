import React from "react";
import { Container } from "@mui/material";
import { BrowserRouter, Route, Routes } from "react-router-dom";

import "./App.css";
import ResponsiveAppBar from "./components/ResponsiveAppBar";
import Home from "./tabs/Home";
import Data from "./tabs/Data";
import Results from "./tabs/Results";
import Play from "./tabs/Play";
import Experiment from "./tabs/Experiment";

function App() {
  return (
    <BrowserRouter>
      <ResponsiveAppBar />
      <Container maxWidth="lg" sx={{ my: 5, mb: 4 }}>
        <Routes>
          <Route path="/" element={<Home />}></Route>
          <Route path="/app" element={<Home />} />
          <Route path="/app/data/" element={<Data />} />
          <Route path="/app/experiments" element={<Experiment />} />
          <Route path="/app/results" element={<Results />} />
          <Route path="/app/play" element={<Play />} />
        </Routes>
      </Container>
    </BrowserRouter>
  );
}
export default App;
