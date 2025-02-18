# Movie Review Sentiment Analysis

This project is a sentiment analysis web application that allows users to input movie reviews and receive sentiment analysis results. The application consists of a **React-based frontend** and a **Python Flask-based backend**.

## How to Start the React Frontend

1. **Navigate to the frontend directory:**
   ```sh
   cd react-frontend
   ```

2. **Install dependencies:**
   ```sh
   npm install
   ```

3. **Start the development server:**
   ```sh
   npm start
   ```

4. **Access the frontend:**
   - Open `http://localhost:3000` in your browser.

## How to Start the Python Backend

1. **Navigate to the backend directory:**
   ```sh
   cd backend
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Flask server:**
   ```sh
   python app.py
   ```

5. **API will be running at:**
   - `http://127.0.0.1:5000`

## Communication Between Frontend and Backend

- The React frontend sends user input (movie reviews) to the Flask backend via an API endpoint.
- The backend processes the text using a sentiment analysis model and returns the sentiment and confidence score.
- The frontend then displays the results to the user.
- The primary API endpoint used:
  - `POST /analyze` - Accepts JSON input `{ "review": "<user input>" }` and returns a JSON response `{ "sentiment": "Positive", "confidence": 0.96 }`.

This structure ensures smooth communication between the frontend and backend, enabling real-time sentiment analysis for user-submitted movie reviews.
