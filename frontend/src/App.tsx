import React from "react";
import logo from "./static/images/logo.png";
import "./App.css";

function App() {
  return (
    <div className="full-body-container">
      <img className="logo" src={logo} alt="Fiend for Beans logo" />{" "}
      <label>Enter which flavor profiles you enjoy.</label>
      <div className="input-box">
        <div>
          <input placeholder="citrus, floral, sweet" id="filter-text-val" />
        </div>
      </div>
      <label>Select which roast you prefer.</label>
      <div className="dropdown">
        <select name="roast" id="roast_select">
          <option value="Light" selected>
            Light
          </option>
          <option value="Medium-Light">Medium-Light</option>
          <option value="Medium">Medium</option>
          <option value="Medium-Dark">Medium-Dark</option>
          <option value="Dark">Dark</option>
        </select>
      </div>
      <br />
      <button id="search-button">Search</button>
      <div id="answer-box"></div>
    </div>
  );
}

export default App;
