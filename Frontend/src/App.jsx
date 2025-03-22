import { useEffect, useState } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
import React from "react";

function App() {
  // Academic scores
  const [math, setMath] = useState(-1);
  const [Science, setScience] = useState(-1);
  const [Literature, setLiterature] = useState(-1);
  const [ComputerScience, setComputerScience] = useState(-1);
  const [History, setHistory] = useState(-1);
  const [Art, setArt] = useState(-1);
  const [Economics, setEconomics] = useState(-1);
  // Interest scores
  const [Techprefernece, TechsetPrefernece] = useState(-1);
  const [Healthprefernece, HealthsetPrefernece] = useState(-1);
  const [Businessprefernece, BusinesssetPrefernece] = useState(-1);
  const [Artprefernece, ArtsetPrefernece] = useState(-1);
  const [Educationprefernece, EducationsetPrefernece] = useState(-1);
  const [Engineeringprefernece, EngineeringsetPrefernece] = useState(-1);
  // Personality traits (default values)
  const [analytical, setAnalytical] = useState(0.5);
  const [creative, setCreative] = useState(0.5);
  const [social, setSocial] = useState(0.5);
  // Response state
  const [studentId, setStudentId] = useState(
    "STUDENT" + Math.floor(Math.random() * 10000)
  );
  const [recommendation, setRecommendation] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showInfoMessage, setShowInfoMessage] = useState(true);

  // Define the Flask API endpoint
  const API_URL = "http://localhost:5000/predict";

  const handleSubmit = async (e) => {
    e.preventDefault();

    setIsLoading(true);
    setError(null);
    setShowInfoMessage(false);

    try {
      // Format data exactly as specified
      const formData = {
        student_id: studentId,
        academic_scores: {
          Math: parseFloat(math),
          Science: parseFloat(Science),
          Literature: parseFloat(Literature),
          "Computer Science": parseFloat(ComputerScience),
          History: parseFloat(History),
          Art: parseFloat(Art),
          Economics: parseFloat(Economics),
        },
        interest_scores: {
          Technology: parseFloat(Techprefernece),
          Healthcare: parseFloat(Healthprefernece),
          Business: parseFloat(Businessprefernece),
          Arts: parseFloat(Artprefernece),
          Education: parseFloat(Educationprefernece),
          Engineering: parseFloat(Engineeringprefernece),
        },
        personality_traits: {
          Analytical: parseFloat(analytical),
          Creative: parseFloat(creative),
          Social: parseFloat(social),
        },
      };
      // Send data to Flask backend
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      const data = await response.json();
      let str = "";
      // setRecommendation(data.recommendation || "No career path found");
      if (data.recommendation) {
        console.log(data.recommendation);
        const sortedEntries = Object.entries(data.recommendation).sort(
          (a, b) => b[1] - a[1]
        ); // Sorting in descending order
        for (const [key, value] of sortedEntries) {
          console.log(`${key} : ${value}\n`);
          str += `${key} : ${value}\n`;
        }
        setRecommendation(str);
      }
    } catch (err) {
      console.error("Error submitting form:", err);
      setError(err.message || "Failed to get career prediction");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      className="verticalContainer"
      style={{
        width: "100vw",
        maxWidth: "100vw",
        margin: 0,
        padding: "2rem",
        boxSizing: "border-box",
      }}
    >
      <div className="formSection">
        <form onSubmit={handleSubmit}>
          <div className="formGrid">
            <div className="formColumn">
              <h2>Student Information</h2>
              <div className="inputContainer">
                <label>Student ID</label>
                <input
                  type="text"
                  value={studentId}
                  onChange={(e) => setStudentId(e.target.value)}
                />
              </div>

              <h2>Your Academic Scores:</h2>
              <div className="inputContainer">
                <label>Math Score (0-100)</label>
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={math === -1 ? "" : math}
                  onChange={(e) => setMath(e.target.value)}
                />
              </div>
              <div className="inputContainer">
                <label>Science Score (0-100)</label>
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={Science === -1 ? "" : Science}
                  onChange={(e) => setScience(e.target.value)}
                />
              </div>
              <div className="inputContainer">
                <label>Literature Score (0-100)</label>
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={Literature === -1 ? "" : Literature}
                  onChange={(e) => setLiterature(e.target.value)}
                />
              </div>
              <div className="inputContainer">
                <label>Computer Science Score (0-100)</label>
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={ComputerScience === -1 ? "" : ComputerScience}
                  onChange={(e) => setComputerScience(e.target.value)}
                />
              </div>
              <div className="inputContainer">
                <label>History Score (0-100)</label>
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={History === -1 ? "" : History}
                  onChange={(e) => setHistory(e.target.value)}
                />
              </div>
              <div className="inputContainer">
                <label>Art Score (0-100)</label>
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={Art === -1 ? "" : Art}
                  onChange={(e) => setArt(e.target.value)}
                />
              </div>
              <div className="inputContainer">
                <label>Economics Score (0-100)</label>
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={Economics === -1 ? "" : Economics}
                  onChange={(e) => setEconomics(e.target.value)}
                />
              </div>
            </div>

            <div className="formColumn">
              <h2>Your Interest Scores (0-10):</h2>
              <div className="inputContainer">
                <label>
                  Technology: {Techprefernece === -1 ? 0 : Techprefernece}
                </label>
                <input
                  type="range"
                  min={0}
                  max={10}
                  value={Techprefernece === -1 ? 0 : Techprefernece}
                  onChange={(e) => TechsetPrefernece(e.target.value)}
                  step={0.1}
                />
              </div>
              <div className="inputContainer">
                <label>
                  Business: {Businessprefernece === -1 ? 0 : Businessprefernece}
                </label>
                <input
                  type="range"
                  min={0}
                  max={10}
                  value={Businessprefernece === -1 ? 0 : Businessprefernece}
                  onChange={(e) => BusinesssetPrefernece(e.target.value)}
                  step={0.1}
                />
              </div>
              <div className="inputContainer">
                <label>
                  Healthcare: {Healthprefernece === -1 ? 0 : Healthprefernece}
                </label>
                <input
                  type="range"
                  min={0}
                  max={10}
                  value={Healthprefernece === -1 ? 0 : Healthprefernece}
                  onChange={(e) => HealthsetPrefernece(e.target.value)}
                  step={0.1}
                />
              </div>
              <div className="inputContainer">
                <label>Arts: {Artprefernece === -1 ? 0 : Artprefernece}</label>
                <input
                  type="range"
                  min={0}
                  max={10}
                  value={Artprefernece === -1 ? 0 : Artprefernece}
                  onChange={(e) => ArtsetPrefernece(e.target.value)}
                  step={0.1}
                />
              </div>
              <div className="inputContainer">
                <label>
                  Education:{" "}
                  {Educationprefernece === -1 ? 0 : Educationprefernece}
                </label>
                <input
                  type="range"
                  min={0}
                  max={10}
                  value={Educationprefernece === -1 ? 0 : Educationprefernece}
                  onChange={(e) => EducationsetPrefernece(e.target.value)}
                  step={0.1}
                />
              </div>
              <div className="inputContainer">
                <label>
                  Engineering:{" "}
                  {Engineeringprefernece === -1 ? 0 : Engineeringprefernece}
                </label>
                <input
                  type="range"
                  min={0}
                  max={10}
                  value={
                    Engineeringprefernece === -1 ? 0 : Engineeringprefernece
                  }
                  onChange={(e) => EngineeringsetPrefernece(e.target.value)}
                  step={0.1}
                />
              </div>

              <h2>Your Personality Traits (0-1):</h2>
              <div className="inputContainer">
                <label>Analytical: {analytical}</label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  value={analytical}
                  onChange={(e) => setAnalytical(e.target.value)}
                  step={0.1}
                />
              </div>
              <div className="inputContainer">
                <label>Creative: {creative}</label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  value={creative}
                  onChange={(e) => setCreative(e.target.value)}
                  step={0.1}
                />
              </div>
              <div className="inputContainer">
                <label>Social: {social}</label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  value={social}
                  onChange={(e) => setSocial(e.target.value)}
                  step={0.1}
                />
              </div>
            </div>
          </div>

          <div className="buttonContainer">
            <button type="submit" disabled={isLoading}>
              {isLoading
                ? "Processing ML Model..."
                : "Generate Career Recommendation"}
            </button>
          </div>
        </form>
      </div>

      <div className="outputSection">
        <h2>Career Recommendation:</h2>
        {showInfoMessage && (
          <div className="info-message">
            <p>
              Fill in your academic scores, interests, and personality traits,
              then click "Generate Career Recommendation" to get AI-powered
              career guidance.
            </p>
            <p>
              The system will generate 2000 synthetic students to train the ML
              model for comparison.
            </p>
          </div>
        )}
        {isLoading && (
          <p className="loading">
            Training ML model with 2000 synthetic students...
          </p>
        )}
        {error && <p className="error">{error}</p>}
        {recommendation && (
          <div className="result">
            <pre>{recommendation}</pre>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
