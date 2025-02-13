import React, { useState } from "react";
import "./App.css";

function App() {
    const [inputText, setInputText] = useState("");
    const [result, setResult] = useState(null);

    // Handles input change
    const handleInputChange = (event) => {
        setInputText(event.target.value);
    };

    // Handles form submission and sends data to backend
    const handleSubmit = async (event) => {
        event.preventDefault();

        // Do not send a request if input is empty
        if (inputText.trim() === "") {
            setResult(null);
            return;
        }

        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: inputText })
        });

        const data = await response.json();
        setResult(data);
    };

    return (
        <div className="app-container">
            {/* Header Section */}
            <header className="header">
                Movie Review Sentiment Analysis
            </header>

            {/* Main Content */}
            <div className="form-wrapper">
                <div className="form-container">
                    <p className="description">
                        Enter a movie review below, and we will analyze whether it is <b>Positive</b> or <b>Negative</b>!
                    </p>
                    <form onSubmit={handleSubmit}>
                        <textarea
                            placeholder="Type your movie review here..."
                            value={inputText}
                            onChange={handleInputChange}
                        />
                        <button type="submit">Analyze Sentiment</button>
                    </form>

                    {/* Result Box (Only show if there is a valid result) */}
                    {result && result.sentiment && (
                        <div className="result-box">
                            <h3>Sentiment: {result.sentiment}</h3>
                            <p>Confidence Score: {result.confidence.toFixed(2)}</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default App;
