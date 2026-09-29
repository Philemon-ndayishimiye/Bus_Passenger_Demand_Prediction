# Bus Passenger Demand Prediction using BiLSTM

A deep-learning system that predicts how many passengers will be waiting at a bus park in a given 30-minute time slot, and from that prediction tells a bus transport company **how many buses it should dispatch**.

---

## 1. The Problem

In Rwanda's bus parks, the number of buses available rarely matches the number of passengers waiting:

- **At peak hours** (early morning and evening), there are too few buses. Passengers queue for a long time, overcrowd the buses that do arrive, and lose productive time.
- **At off-peak hours**, there are too many buses. They wait half-empty or leave with very few passengers, wasting fuel, driver time and money for the transport company.

Today, dispatch decisions are mostly made by experience and guesswork. Companies have no reliable tool that looks at the time of day, the day of the week, whether it is a holiday, and how many passengers the previous buses carried, and then says: _"For this slot you need this many buses."_

## 2. Project Aim

The project aims to help bus transport companies **predict the number of buses needed** for each time slot and route, so that supply matches demand.

Specific objectives:

1. Build a dataset of passenger counts per 30-minute time slot for routes leaving a central bus park.
2. Train a **Bidirectional LSTM (BiLSTM)** neural network that predicts the passenger count for a time slot using time features and the passenger counts of the previous buses.
3. Convert the predicted passenger count into the **number of buses required** (based on bus capacity) and a **demand level** (Low / Medium / High).
4. Provide a short-term **multi-step forecast** (+15, +30, +45, +60 minutes) so dispatchers can plan ahead.
5. Deliver the model through a **Streamlit dashboard** and a secured **Flask REST API** with user accounts and saved prediction history (PostgreSQL).

## 3. Methodology

### 3.1 Dataset

File: `passenger_count_dataset.csv` **64,260 records, 11 columns, no missing values.**

| Column            | Description                                         |
| ----------------- | --------------------------------------------------- |
| `Time`            | 30-minute slot, e.g. `05:00-05:30` (05:00 to 22:00) |
| `Time of Day`     | Morning / Afternoon / Evening                       |
| `Day of Week`     | Monday … Sunday                                     |
| `Peak/Off-Peak`   | Whether the slot is a peak period                   |
| `Holiday/Event`   | Yes / No                                            |
| `Bus Capacity`    | Seats per bus (70)                                  |
| `Passenger Count` | **Target** — passengers in that slot                |
| `BusStop`         | Departure bus park (Downtown)                       |
| `Destination`     | Batsinda, Kimironko, Nyamirambo                     |
| `Lag t-1`         | Passenger count of the previous slot                |
| `Lag t-2`         | Passenger count two slots before                    |

Passenger counts range from 12 to 255 (mean ≈ 106).

### 3.2 Pre-processing

1. **Time parsing** the slot start time is converted to a numeric hour (e.g. `07:30-08:00` → `7.5`).
2. **Label encoding**: `Time of Day`, `Day of Week`, `BusStop` and `Destination` are encoded as integers (encoders saved as `le_*.pkl`). Peak and Holiday are encoded as 0/1.
3. **Feature set (10 features):** `hour_numeric, time_of_day_enc, day_of_week_enc, peak_enc, holiday_enc, Bus Capacity, busstop_enc, destination_enc, Lag t-1, Lag t-2`.
4. **Scaling**: features and target are scaled with separate scalers (`scaler_X.pkl`, `scaler_y.pkl`).
5. **Sequence building**: sliding windows of **10 consecutive time steps** are created, giving input of shape `(samples, 10, 10)`.
6. **Train/test split**: 80% / 20% **without shuffling**, so the temporal order is preserved (51,400 training and 12,850 test sequences).

### 3.3 Model Architecture BiLSTM

A Bidirectional LSTM reads each sequence both forwards and backwards, which lets it capture demand patterns before and after a given point in the sequence.

```
Input (10 timesteps × 10 features)
 - Bidirectional LSTM(128) + BatchNorm + Dropout(0.3)
 - Bidirectional LSTM(64)  + BatchNorm + Dropout(0.3)
 - Bidirectional LSTM(32)  + BatchNorm + Dropout(0.2)
 - Dense(64, ReLU) + Dropout(0.2)
 - Dense(32, ReLU)
 - Dense(1)  → predicted passenger count
```

- Total parameters: **355,969**
- Optimizer: Adam (learning rate 0.001), loss: Huber, metric: MAE
- Training: up to 60 epochs, batch size 64, 20% validation split
- Callbacks: EarlyStopping (patience 15), ModelCheckpoint (best model → `best_bilstm_model.h5`), ReduceLROnPlateau

### 3.4 From Prediction to Decision

```
buses_required = ceil(predicted_passengers / bus_capacity)

Demand level:  ≤ 50 → Low    |   51–100 → Medium   |   > 100 → High
```

For the **multi-step forecast**, the model's prediction is fed back as the new `Lag t-1` (and the old `Lag t-1` becomes `Lag t-2`), and the model predicts the next interval, repeated 4 times.

### 3.5 Evaluation Results (test set, 12,850 sequences)

**Regression metrics (passenger count):**

| Metric | Value                |
| ------ | -------------------- |
| MAE    | **17.35 passengers** |
| RMSE   | **21.07 passengers** |
| R²     | **0.7772**           |

**Classification metrics (demand level Low / Medium / High):**

| Metric               | Value      |
| -------------------- | ---------- |
| Accuracy             | **80.11%** |
| Precision (weighted) | 0.7607     |
| Recall (weighted)    | 0.8011     |
| F1-score (weighted)  | 0.7739     |

Generated charts: `training_history.png`, `scatter_actual_vs_predicted.png`, `confusion_matrix.png`, `timeseries_forecast.png`.

> **Limitation:** the model performs well on Medium and High demand but does not correctly identify the **Low** demand class (only 885 of 12,850 test samples are Low). Balancing the data or weighting the Low class is recommended as future work.

## 4. Project Structure

```
folder_name/
├── passenger_count_model.ipynb     # Data exploration, training, evaluation, export
├── passenger_count_dataset.csv     # Dataset (64,260 rows)
│
├── best_bilstm_model.h5            # Trained BiLSTM model
├── scaler_X.pkl / scaler_y.pkl     # Feature and target scalers
├── le_tod.pkl / le_dow.pkl         # Label encoders (time of day, day of week)
├── le_stop.pkl / le_dest.pkl       # Label encoders (bus stop, destination)
├── model_config.json               # Feature list, sequence length, valid classes
│
├── app.py                          # Streamlit dashboard (no database needed)
│
├── api.py                          # Flask REST API (main entry point)
├── auth.py                         # /auth — register, login, logout, profile
├── users.py                        # /users — admin user management
├── predictions.py                  # /predict, /predictions, /predictions/stats
├── models.py                       # Database tables (User, Prediction, TokenBlocklist)
├── database.py                     # PostgreSQL connection + table creation
├── seed.py                         # Creates the default admin user
├── migrate.py                      # Adds the "role" column to an old users table
├── requirements.txt
├── .env                            # Environment variables (do NOT commit)
└── *.png                           # Evaluation charts
```

## 5. How to Run the Project (Step by Step)

There are **two ways** to use the model:

- **Option A Streamlit dashboard:** the quickest way to see predictions. No database needed.
- **Option B Flask REST API:** the full backend with login, PostgreSQL and saved prediction history (used by the web frontend).

### Step 1 Install the prerequisites

- **Python 3.10 – 3.12** (the model was trained with TensorFlow 2.20)
- **PostgreSQL 14+** (only for Option B)
- Git (optional)

Check your Python version:

```bash
python --version
```

### Step 2 Open the project folder

```bash
cd folder_name
```

### Step 3 Create and activate a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4 Fix `requirements.txt` and install the libraries

In `requirements.txt`, the line `flask-jwt-extended==4.6.0,` has a trailing comma, which makes `pip` fail. Change it to:

```
flask-jwt-extended==4.6.0
```

Then install:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5 (optional) Retrain the model

The trained model files are already included, so you can skip this step. To retrain:

```bash
pip install jupyter
jupyter notebook passenger_count_model.ipynb
```

Run all cells from top to bottom (**Kernel Restart & Run All**). This regenerates `best_bilstm_model.h5`, the scalers, the encoders, `model_config.json` and all charts.

---

### Option A Run the Streamlit Dashboard

### Step 6A Start the app

```bash
streamlit run app.py
```

Your browser opens at **http://localhost:8501**.

### Step 7A Make a prediction

1. Choose the **time slot**, **day of week**, **time of day** and **peak / off-peak**.
2. Choose the **bus stop** (Downtown) and the **destination** (Batsinda, Kimironko or Nyamirambo).
3. Choose **holiday** Yes/No and set the **bus capacity** (default 70).
4. Enter the passenger counts of the **last two buses** (`Lag t-1`, `Lag t-2`).
5. Click **Predict Now**.

The app shows the predicted passengers, the **number of buses required**, the demand level and a summary table.

---

### Option B Run the Flask REST API

### Step 6B Create the PostgreSQL database

Open `psql` (or pgAdmin) and run:

```sql
CREATE DATABASE bus_prediction;
```

### Step 7B Configure the `.env` file

Create or edit the `.env` file in the project folder:

```env
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/bus_prediction
JWT_SECRET_KEY=put-a-long-random-secret-string-here
DEFAULT_USER_EMAIL=admin@example.com
DEFAULT_USER_PASSWORD=Admin@1234
DEFAULT_USER_NAME=Admin
```

Replace `YOUR_PASSWORD` with your PostgreSQL password.

### Step 8B Create the default admin user

```bash
python seed.py
```

This creates the database tables and an admin account using the credentials in `.env`.

> `migrate.py` is only needed if you have an **old** database whose `users` table has no `role` column. Before running it, change the hard-coded email inside the file to your admin email.

### Step 9B Start the API

```bash
python api.py
```

The API runs at **http://localhost:5000**. Opening that address shows a health message and the list of endpoints.

### Step 10B: Test the API

**1. Log in and get a token**

```bash
curl -X POST http://localhost:5000/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"email\": \"admin@example.com\", \"password\": \"Admin@1234\"}"
```

Copy the `access_token` from the response.

**2. Make a prediction**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "{\"time_slot\": \"07:00-07:30\", \"time_of_day\": \"Morning\", \"day_of_week\": \"Monday\", \"peak_status\": \"Peak\", \"holiday\": \"No\", \"bus_stop\": \"Downtown\", \"destination\": \"Kimironko\", \"bus_capacity\": 70, \"lag_t1\": 120, \"lag_t2\": 110}"
```

The response contains the predicted passengers, buses required, utilisation, demand level and remaining capacity. The prediction is also saved in the database.

### API Endpoints

| Method     | Endpoint             | Auth    | Description                                             |
| ---------- | -------------------- | ------- | ------------------------------------------------------- |
| GET        | `/`                  | –       | Health check and endpoint list                          |
| GET        | `/config`            | –       | Valid values for dropdowns (slots, days, destinations…) |
| POST       | `/auth/register`     | –       | Create an account                                       |
| POST       | `/auth/login`        | –       | Log in and receive a JWT token                          |
| GET        | `/auth/me`           | ✔       | Current user profile                                    |
| POST       | `/auth/logout`       | ✔       | Log out (token revoked)                                 |
| POST       | `/predict`           | ✔       | Predict passengers and buses for one slot               |
| POST       | `/forecast`          | ✔       | Forecast for +15, +30, +45, +60 minutes                 |
| GET        | `/predictions`       | ✔       | Prediction history (admins see all users)               |
| GET        | `/predictions/stats` | ✔       | Dashboard statistics                                    |
| GET / POST | `/users`             | ✔ admin | List or create users                                    |
| DELETE     | `/users/<id>`        | ✔ admin | Delete a user                                           |
| GET        | `/users/stats`       | ✔ admin | User statistics                                         |

✔ = send the header `Authorization: Bearer <token>`.

> CORS is configured for a frontend running at `http://localhost:5173` (Vite / React). If your frontend runs on another address, update the origin in `api.py`.

## 6. Troubleshooting

| Problem                                                    | Solution                                                                        |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `ERROR: Invalid requirement: 'flask-jwt-extended==4.6.0,'` | Remove the trailing comma in `requirements.txt` (Step 4).                       |
| `DATABASE_URL is not set in your .env file!`               | Create the `.env` file in the same folder as `api.py` (Step 7B).                |
| `password authentication failed`                           | Check the username and password in `DATABASE_URL`.                              |
| Error loading `best_bilstm_model.h5`                       | Use a TensorFlow version close to 2.20, or retrain the model with the notebook. |
| `FileNotFoundError` for a `.pkl` file                      | Run the app from inside the `celine/` folder so it finds the model files.       |
| `401 Missing Authorization Header`                         | Log in first and send the token in the `Authorization` header.                  |

## 7. Future Work

- Collect real passenger counts from more bus parks and routes in Kigali and upcountry.
- Balance the dataset so the **Low** demand class is predicted correctly.
- Add weather, school calendar and special events as extra features.
- Connect the system to live ticketing data for real-time automatic dispatch.

## 8. Technologies Used

Python · TensorFlow / Keras · scikit-learn · pandas · NumPy · Matplotlib · Seaborn · Streamlit · Flask · Flask-JWT-Extended · SQLAlchemy · PostgreSQL · bcrypt
