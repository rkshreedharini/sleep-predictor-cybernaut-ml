import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# STEP 1: CREATE TRAINING DATA
# -------------------------------
np.random.seed(42)
rows = 700

df = pd.DataFrame({
    "sleep_duration": np.random.uniform(4, 9, rows),
    "bedtime": np.random.randint(20, 24, rows),
    "wake_time": np.random.randint(5, 9, rows),
    "caffeine": np.random.choice(["None", "Low", "Moderate", "High"], rows),
    "exercise": np.random.randint(0, 90, rows),
    "screen_time": np.random.randint(0, 180, rows),
    "stress": np.random.randint(1, 11, rows),
    "mood": np.random.choice(["Happy", "Neutral", "Sad", "Anxious"], rows),
    "interruptions": np.random.choice([0, 1], rows)
})

def label_sleep(row):
    if row.sleep_duration >= 7 and row.screen_time < 60 and row.stress <= 4 and row.interruptions == 0:
        return "Good"
    elif row.sleep_duration >= 6:
        return "Average"
    return "Poor"

df["sleep_quality"] = df.apply(label_sleep, axis=1)

# -------------------------------
# STEP 2: ENCODING
# -------------------------------
encoders = {}
for col in ["caffeine", "mood", "sleep_quality"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("sleep_quality", axis=1)
y = df["sleep_quality"]

# -------------------------------
# STEP 3: TRAIN MODEL
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=250, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# INPUT UTILITIES
# -------------------------------
def safe_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("❌ Enter a valid number")

def parse_time(prompt):
    while True:
        value = input(prompt).strip().lower()
        try:
            if "am" in value or "pm" in value:
                value = value.replace(" ", "")
                return datetime.strptime(value, "%I%p").hour
            if ":" in value:
                return int(datetime.strptime(value, "%H:%M").hour)
            hour = int(value)
            if 0 <= hour <= 23:
                return hour
            print("❌ Hour must be between 0 and 23")
        except ValueError:
            print("❌ Examples: 10 pm, 6 am, 22, 22:30")

def calculate_sleep_duration(bed, wake):
    duration = (wake - bed) % 24
    if duration == 0 or duration > 14:
        print("⚠️ Unusual sleep duration detected, setting to 8 hours")
        return 8.0
    return float(duration)

# -------------------------------
# USER INPUT & HISTORY
# -------------------------------
history = []

def user_input():
    print("\n--- Sleep Quality Predictor ---")

    bedtime = parse_time("Bedtime (e.g., 10 pm, 22, 22:30): ")
    wake_time = parse_time("Wake-up Time (e.g., 6 am, 6, 06:30): ")

    sleep_duration = calculate_sleep_duration(bedtime, wake_time)
    print(f"🕒 Auto-calculated Sleep Duration: {sleep_duration:.1f} hours")

    caffeine_map = {"none":0, "low":1, "moderate":2, "high":3}
    while True:
        caffeine = input("Caffeine Intake (None/Low/Moderate/High): ").lower()
        if caffeine in caffeine_map:
            caffeine = caffeine_map[caffeine]
            break
        print("❌ Invalid option")

    exercise = safe_int("Exercise Duration (minutes): ")
    screen_time = safe_int("Screen Time Before Bed (minutes): ")

    while True:
        stress = safe_int("Stress Level (1-10): ")
        if 1 <= stress <= 10:
            break
        print("❌ Must be between 1 and 10")

    mood_map = {"happy":0, "neutral":1, "sad":2, "anxious":3}
    while True:
        mood = input("Mood Before Sleep (Happy/Neutral/Sad/Anxious): ").lower()
        if mood in mood_map:
            mood = mood_map[mood]
            break
        print("❌ Invalid mood")

    interruptions = safe_int("Sleep Interruptions? (0 = No, 1 = Yes): ")

    return pd.DataFrame([{
        "sleep_duration": sleep_duration,
        "bedtime": bedtime,
        "wake_time": wake_time,
        "caffeine": caffeine,
        "exercise": exercise,
        "screen_time": screen_time,
        "stress": stress,
        "mood": mood,
        "interruptions": interruptions
    }]), sleep_duration

# -------------------------------
# PREDICTION
# -------------------------------
def predict():
    data, duration = user_input()
    pred = model.predict(data)[0]
    result = encoders["sleep_quality"].inverse_transform([pred])[0]

    history.append({
        "day": len(history) + 1,
        "sleep_duration": duration,
        "sleep_quality": result
    })

    print(f"\n🛌 Predicted Sleep Quality: {result}")

    print("\n💡 Suggestions:")
    if result == "Poor":
        print("- Reduce screen time before bed")
        print("- Avoid caffeine at night")
        print("- Try relaxation techniques")
    elif result == "Average":
        print("- Increase sleep duration slightly")
        print("- Maintain consistent bedtime")
    else:
        print("- Excellent sleep habits 🌙 Keep it up!")

# -------------------------------
# GRAPH FUNCTION
# -------------------------------
def show_sleep_graph():
    if not history:
        print("\nNo data to plot yet.")
        return

    days = [h["day"] for h in history]
    durations = [h["sleep_duration"] for h in history]

    quality_map = {"Poor": 1, "Average": 2, "Good": 3}
    quality_values = [quality_map[h["sleep_quality"]] for h in history]

    plt.figure()
    plt.plot(days, durations, marker='o', label="Sleep Duration (hrs)")
    plt.plot(days, quality_values, marker='s', label="Sleep Quality Level")

    plt.xlabel("Day")
    plt.ylabel("Value")
    plt.title("Sleep Duration & Quality Over Time")
    plt.yticks([1, 2, 3], ["Poor", "Average", "Good"])
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    print("\nOptions:")
    print("[1] New Sleep Entry")
    print("[2] View Sleep Graph")
    print("[3] Exit")

    choice = input("Choose option: ")

    if choice == "1":
        predict()
    elif choice == "2":
        show_sleep_graph()
    elif choice == "3":
        print("\nGood night 🌙 Sleep well!")
        break
    else:
        print("❌ Invalid option")
