import pandas as pd
import os
import io
import random
import requests
import tensorflow as tf
import numpy as np
import seaborn as sns
import logging
from sklearn.preprocessing import MinMaxScaler
import joblib
import plotly.express as px
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
from flask import Flask, render_template,  jsonify, request, redirect, url_for, session,flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from models import users, User

Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Bidirectional = tf.keras.layers.Bidirectional
l2 = tf.keras.regularizers.l2
EarlyStopping = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau

app = Flask(__name__)

API_URL = "https://ourworldindata.org/grapher/co2-fossil-plus-land-use.csv?v=1&csvType=full&useColumnShortNames=true"
LOCAL_FILE = "co2-fossil-land-use-all.csv"
MODEL_FILE = "model_lstm.h5"
LOG_FILE = "update_log.txt" 
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.secret_key = "secretkey"

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

try:
    model_user = tf.keras.models.load_model('model_lstm.h5')
    scaler = joblib.load("scaler.pkl")
    print("Model dan Scaler berhasil dimuat.")
except Exception as e:
    print(f"Error loading model/scaler: {e}")

df = pd.read_csv(LOCAL_FILE)

df = df[df["Entity"] == "Indonesia"].copy()
df = df.dropna(subset=['Year', 'emissions_total_including_land_use_change'])
data = df[['Year', 'emissions_total_including_land_use_change']].sort_values('Year')
data_actual = df[['Year', 'emissions_total_including_land_use_change']].sort_values('Year')

def iqr_outliers(data):
    outliers_indices = {}
    numeric_df = data.select_dtypes(include=['float64', 'int64'])
    q1 = numeric_df.quantile(0.25)
    q3 = numeric_df.quantile(0.75)
    iqr = q3 - q1
    lower_tail = q1 - 1.5 * iqr
    upper_tail = q3 + 1.5 * iqr
    for column in numeric_df.columns:
        outliers = numeric_df[(numeric_df[column] < lower_tail[column]) | (numeric_df[column] > upper_tail[column])].index
        outliers_indices[column] = list(outliers)
    return outliers_indices

def winsorize(data, cols, limits):
    for col in cols:
        q1, q3 = data[col].quantile([0.25, 0.75])
        iqr_val = q3 - q1 
        lower_bound = q1 - limits * iqr_val
        upper_bound = q3 + limits * iqr_val
        data[col] = np.clip(data[col], lower_bound, upper_bound)  # Menggunakan np.clip untuk memotong nilai yang di luar batas
    return data

num_cols = [
    'emissions_total_including_land_use_change'
]

data = winsorize(data.copy(), num_cols, 1.5)

# Normalisasi data
scaled_data = scaler.transform(data['emissions_total_including_land_use_change'].values.reshape(-1, 1))

# Fungsi untuk membuat dataset dengan time_steps
def create_dataset(dataset, time_steps=10):
    X = []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:(i + time_steps), 0])
    return np.array(X)

# Buat input untuk prediksi historis
time_steps = 10
X_input = create_dataset(scaled_data, time_steps)
X_input = X_input.reshape(X_input.shape[0], X_input.shape[1], 1)

# Prediksi menggunakan model
predicted_values = model_user.predict(X_input)
predicted_values = scaler.inverse_transform(predicted_values)

# Pastikan panjangnya sama dengan jumlah tahun
predicted_years = data['Year'].iloc[time_steps:].values
actual_values = data['emissions_total_including_land_use_change'].iloc[time_steps:].values

# Simpan hasil dalam DataFrame
prediksi_df = pd.DataFrame({
    'Year': predicted_years,
    'Actual CO₂ Emissions': actual_values,
    'Predicted CO₂ Emissions': predicted_values.flatten()
})

# Prediksi 5 tahun ke depan
future_steps = 5
last_inputs = scaled_data[-time_steps:].reshape(1, time_steps, 1)

future_predictions = []
last_inputs = scaled_data[-time_steps:].reshape(1, time_steps, 1)

print("Last inputs sebelum prediksi:", last_inputs.flatten())

for i in range(future_steps):
    next_pred = model_user.predict(last_inputs, verbose=0)[0, 0]
    future_predictions.append(next_pred)

    print(f"Prediksi {i+1}: {next_pred}") 

    last_inputs = np.append(last_inputs[:, 1:, :], [[[next_pred]]], axis=1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

print("Prediksi setelah inverse transform:", future_predictions.flatten())

last_year = int(data['Year'].iloc[-1])
future_years = [last_year + i for i in range(1, future_steps + 1)]

future_df = pd.DataFrame({
    'Year': future_years,
    'Predicted Future CO₂ Emissions': future_predictions.ravel()  # Pastikan 1D
})

def create_interactive_plot():

    fig = go.Figure()


    fig.add_trace(go.Scatter(
        x=data_actual['Year'],
        y=data_actual['emissions_total_including_land_use_change'],
        mode='lines+markers',
        name='Actual CO₂ Emissions',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=prediksi_df['Year'],
        y=prediksi_df['Predicted CO₂ Emissions'],
        mode='lines+markers',
        name='Predicted CO₂ Emissions',
        line=dict(color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=future_df['Year'],
        y=future_df['Predicted Future CO₂ Emissions'],
        mode='lines+markers',
        name='Future Predictions (5 Years)',
        line=dict(color='green', dash='dot')
    ))

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="CO₂ Emissions",
        legend=dict(x=0.01, y=0.99, font=dict(size=12)),
        template="plotly_white"
    )

    
    pio.write_html(fig, "static/predict_line.html", config={'displayModeBar': False})

def generate_plot(future_years, future_predictions):
    """ Membuat grafik interaktif """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Year'], y=scaler.inverse_transform(data_actual[['emissions_total_including_land_use_change']]).flatten(),
                             mode='lines', name='Data Aktual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_years, y=future_predictions, mode='markers+lines',
                             name='Prediksi', marker=dict(color='red', size=8)))
    fig.update_layout(
        xaxis_title="Tahun", 
        yaxis_title="Emisi CO₂ (Miliar Ton)", 
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, font=dict(size=12), 
        bgcolor="rgba(255,255,255,0.7)"))
    return pio.to_html(fig, full_html=False, config={'displayModeBar': False})

# ---------------------------
# Load Global Data for Visualization
# ---------------------------
df_global = pd.read_csv(LOCAL_FILE)

def prepare_data(df_global):
    latest_year = df_global['Year'].max()

    df_latest = df_global[df_global['Year'] == latest_year]

    return df_latest


def create_map(df_global):
    """ Membuat peta interaktif dengan Plotly """
    df = prepare_data(df_global) 

    color_scale = [
        (0.00, "lightcyan"),    
        (0.001, "#ffdcb1"),  
        (0.02, "#ffc278"),       
        (0.05, "#ffa93f"),       
        (0.10, "#ff8f06"),   
        (0.30, "#cc7000"),
        (0.40, "#935100"),
        (1.00, "#834800")         
    ]

    fig = px.choropleth(df,
                        locations="Code",  
                        color="emissions_total_including_land_use_change",
                        hover_name="Entity",  
                        color_continuous_scale=color_scale,
                        title=f"Annual CO₂ Emissions Including Land-Use Change, {df_global['Year'].max()}",
                        labels={'emissions_total_including_land_use_change': 'CO₂ Emissions (Million Tonnes)'},
                        locationmode="ISO-3"
                        )

 
    fig.update_layout(
        geo=dict(showcoastlines=True),
        coloraxis_colorbar=dict(
            title="CO₂ Emissions",
            orientation='h',  
            thicknessmode="pixels", thickness=10,  
            lenmode="pixels", len=400,  
            x=0.5,  
            y=-0.2,  
            xanchor='center',
            yanchor='bottom'
        )
    )

 
    if not os.path.exists("static"):
        os.makedirs("static")
    
    pio.write_html(fig, "static/map.html")


def create_lineChart(df_global):
    df = df_global[df_global["Entity"] == "Indonesia"].dropna(subset=['Year', 'emissions_total_including_land_use_change'])
    df = df.sort_values(by="Year", ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Year"], 
        y=df["emissions_total_including_land_use_change"], 
        mode="lines+markers", 
        name="Total Emisi",
        line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=df["Year"], 
        y=df["emissions_from_land_use_change"], 
        mode="lines+markers", 
        name="From land-use change",
        line=dict(color="green")
    ))

    fig.add_trace(go.Scatter(
        x=df["Year"], 
        y=df["emissions_total"], 
        mode="lines+markers", 
        name="From fossil fuels",
        line=dict(color="red")
    ))


    fig.update_layout(
        title="Tren Emisi CO₂ Indonesia",
        xaxis_title="Tahun",
        yaxis_title="Emisi CO₂ (Juta Ton)",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    pio.write_html(fig, "static/line_chart.html", config={'displayModeBar': False})

def barChart5(df_global):
    latest_year = df_global["Year"].max()


    df_last = df_global[
        (df_global["Year"] == latest_year) & 
        (df_global["Code"].str.len() == 3) 
    ]


    df_top10 = df_last.nlargest(10, "emissions_total_including_land_use_change")

    fig = px.bar(
        df_top10, 
        x="emissions_total_including_land_use_change", 
        y="Entity", 
        orientation="h",
        text="emissions_total_including_land_use_change",
        labels={"emissions_total_including_land_use_change": "Total Emission (Juta Ton)", "Entity": "Negara"},
    )

    fig.update_traces(texttemplate="%{text}", textposition="outside")
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), plot_bgcolor="rgba(0,0,0,0)",  # Hilangkan background dalam chart
    paper_bgcolor="rgba(0,0,0,0)",)

 
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"), 
        xaxis=dict(title="Emisi CO₂ (Juta Ton)", tickformat=",d"),
        plot_bgcolor="rgba(0,0,0,0)",  
        paper_bgcolor="rgba(0,0,0,0)",
    )


    pio.write_html(fig, "static/bar_chart_5.html", config={'displayModeBar': False})




def barChartLand(df_global):
    latest_year = df_global["Year"].max()

   
    df_land = df_global[
        (df_global["Year"] == latest_year) & 
        (df_global["Code"].str.len() == 3)  
    ]

    df_top10_land= df_land.sort_values(by="emissions_from_land_use_change", ascending=False).head(10)
    fig = px.bar(
        df_top10_land, 
        x="emissions_from_land_use_change", 
        y="Entity", 
        orientation="h",
        text="emissions_from_land_use_change",
        labels={"emissions_from_land_use_change": "Land-use (Juta Ton)", "Entity": "Negara"},
    )

    fig.update_traces(texttemplate="%{text}", textposition="outside")
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), plot_bgcolor="rgba(0,0,0,0)",  # Hilangkan background dalam chart
    paper_bgcolor="rgba(0,0,0,0)",)

    pio.write_html(fig, "static/bar_chart_land.html", config={'displayModeBar': False})

#Barchart Fossil
def barChartFossil(df_global):
    latest_year = df_global["Year"].max()

 
    df_foss = df_global[
        (df_global["Year"] == latest_year) & 
        (df_global["Code"].str.len() == 3)  
    ]


    df_top10 = df_foss.sort_values(by="emissions_total", ascending=False).head(10)
    fig = px.bar(
        df_top10, 
        x="emissions_total", 
        y="Entity", 
        orientation="h",
        text="emissions_total",
        labels={"emissions_total": "Fossil Fuels (Juta Ton)", "Entity": "Negara"},
    )

    fig.update_traces(texttemplate="%{text}", textposition="outside")
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), plot_bgcolor="rgba(0,0,0,0)",  # Hilangkan background dalam chart
    paper_bgcolor="rgba(0,0,0,0)",)

    pio.write_html(fig, "static/bar_chart_fossil.html", config={'displayModeBar': False})


def growtStatus(df_global):
    df_indonesia = df_global[df_global["Entity"] == "Indonesia"].copy()
    df_indonesia = df_indonesia.sort_values(by="Year", ascending=True)
    last_two_years = df_indonesia.tail(2)

    if len(last_two_years) < 2:
        return {"error": "Data tidak cukup untuk menghitung pertumbuhan"}

    emission_prev = last_two_years.iloc[0]["emissions_total_including_land_use_change"]
    emission_latest = last_two_years.iloc[1]["emissions_total_including_land_use_change"]

    land_use_prev = last_two_years.iloc[0]["emissions_from_land_use_change"]
    land_use_latest = last_two_years.iloc[1]["emissions_from_land_use_change"]

    fossil_prev = last_two_years.iloc[0]["emissions_total"]
    fossil_latest = last_two_years.iloc[1]["emissions_total"]

    def calculate_growth(prev, latest):
        if prev == 0:
            return "N/A"
        growth = ((latest - prev) / prev) * 100
        return f"{'+' if growth > 0 else ''}{growth:.2f}%"

    growth_total = calculate_growth(emission_prev, emission_latest)
    growth_land_use = calculate_growth(land_use_prev, land_use_latest)
    growth_fossil = calculate_growth(fossil_prev, fossil_latest)

    latest_year = last_two_years.iloc[1]["Year"]

    return {
        "year": latest_year,
        "growth_total": growth_total,
        "growth_land_use": growth_land_use,
        "growth_fossil": growth_fossil
    }



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_update_log():
    """Menyimpan waktu terakhir update ke dalam file log."""
    with open(LOG_FILE, "w") as log:
        log.write(f"Terakhir diperbarui: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def get_last_update():
    """Mengambil waktu terakhir update dari file log."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as log:
            return log.readline().strip()
    return "Belum ada pembaruan data"

def update_dataset():
    """Mengambil data CO2 terbaru dari API."""
    try:
        logging.info("Mengambil data terbaru dari API...")
        response = requests.get(API_URL, timeout=10)
        response.raise_for_status()
        
        df_api = pd.read_csv(io.StringIO(response.text))
        required_columns = {'Entity', 'Code', 'Year', 'emissions_total_including_land_use_change', 
                            'emissions_from_land_use_change', 'emissions_total'}
        if not required_columns.issubset(df_api.columns):
            logging.error("Kolom yang diperlukan tidak ditemukan dalam data API.")
            return {"status": "error", "message": "Kolom yang diperlukan tidak ditemukan dalam data API"}

        df_global = df_api[list(required_columns)]

        if os.path.exists(LOCAL_FILE):
            df_lama = pd.read_csv(LOCAL_FILE)
            df_updated = pd.concat([df_lama, df_global]).drop_duplicates(subset=['Entity', 'Year']).reset_index(drop=True)
        else:
            df_updated = df_global

        df_updated.to_csv(LOCAL_FILE, index=False)
        save_update_log()
        logging.info("Dataset berhasil diperbarui!")

        return {"status": "success", "message": "Dataset berhasil diperbarui!"}
    except requests.exceptions.RequestException as e:
        logging.error(f"Error API: {e}")
        return {"status": "error", "message": f"Error API: {e}"}
    

def preprocess_data():
    """Preprocessing: menangani outlier dan normalisasi."""
    try:
        df = pd.read_csv(LOCAL_FILE)
        df = df[df["Entity"] == "Indonesia"].copy()
        df = df.dropna(subset=['Year', 'emissions_total_including_land_use_change'])
        data = df[['Year', 'emissions_total_including_land_use_change']].sort_values('Year')

        # Simpan Boxplot Sebelum Outlier Handling
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=data['emissions_total_including_land_use_change'])
        plt.title("Sebelum Outlier Handling")
        plt.savefig("static/images/boxplot_outlier_before.png")
        plt.close()

        # Fungsi Winsorizing
        def winsorize(data, col, limits=1.5):
            data = data.copy()
            q1, q3 = data[col].quantile([0.25, 0.75])
            iqr_val = q3 - q1
            lower_bound = q1 - limits * iqr_val
            upper_bound = q3 + limits * iqr_val
            data[col] = np.clip(data[col], lower_bound, upper_bound)
            return data

        # Terapkan Winsorizing
        data = winsorize(data, 'emissions_total_including_land_use_change')

        # Simpan Boxplot Setelah Outlier Handling
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=data['emissions_total_including_land_use_change'])
        plt.title("Setelah Outlier Handling")
        plt.savefig("static/images/boxplot_outlier_after.png")
        plt.close()

        # Normalisasi
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['emissions_total_including_land_use_change']])

        # Konversi ke JSON
        df_normalization = pd.DataFrame({
            'Year': data['Year'].values,
            'Before Scaling': data['emissions_total_including_land_use_change'].values,
            'After Scaling': scaled_data.flatten()
        })

        return {
            "status": "success",
            "data": df_normalization.to_dict(orient='records'),
            "scaled_data": scaled_data.tolist()  # Konversi ndarray ke list agar bisa dikembalikan sebagai JSON
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
def get_mape_category(mape):
    """Menentukan kategori model berdasarkan nilai MAPE."""
    if mape < 10:
        return "Sangat Baik"
    elif 10 <= mape < 20:
        return "Baik"
    elif 20 <= mape < 50:
        return "Cukup"
    else:
        return "Kurang BaiK"
    
def train_lstm():
    """Melatih ulang model LSTM dengan data terbaru."""
    try:
        if not os.path.exists(LOCAL_FILE):
            return {"status": "error", "message": "Dataset tidak tersedia. Silakan update data terlebih dahulu."}

        seed_value = 42
        np.random.seed(seed_value)
        random.seed(seed_value)
        tf.random.set_seed(seed_value)

        preprocess_result = preprocess_data()
        if preprocess_result["status"] == "error":
            return preprocess_result

        data = preprocess_result["data"]
        scaled_data = preprocess_result["scaled_data"]


        def create_dataset(dataset, time_steps=10):
            dataset = np.array(dataset) 
            X, y = [], []
            for i in range(len(dataset)-time_steps):
                X.append(dataset[i:(i+time_steps), 0]) 
                y.append(dataset[i+time_steps, 0])
            return np.array(X), np.array(y)

        time_steps = 10
        X, y = create_dataset(scaled_data, time_steps)

  
        X = X.reshape(X.shape[0], X.shape[1], 1)


        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]


        model = Sequential([
            LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            LSTM(units=32, kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(units=1, kernel_regularizer=l2(0.001))
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)


        model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), 
                  callbacks=[early_stopping, lr_scheduler], shuffle=False)


        model.save(MODEL_FILE)

        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

    
        train_predict = scaler.inverse_transform(np.column_stack((train_predict, np.zeros_like(train_predict))))[:, 0]
        y_train_actual = scaler.inverse_transform(np.column_stack((y_train, np.zeros_like(y_train))))[:, 0]
        test_predict = scaler.inverse_transform(np.column_stack((test_predict, np.zeros_like(test_predict))))[:, 0]
        y_test_actual = scaler.inverse_transform(np.column_stack((y_test, np.zeros_like(y_test))))[:, 0]

        def mean_absolute_percentage_error(y_true, y_pred): 
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

 
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        mae = mean_absolute_error(y_test_actual.flatten(), test_predict.flatten())
        rmse = np.sqrt(mean_squared_error(y_test_actual.flatten(), test_predict.flatten()))
        mape = mean_absolute_percentage_error(y_test_actual.flatten(), test_predict.flatten())

        model_category = get_mape_category(mape)

        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Kategori Model: {model_category}")

        return {
            "status": "success",
            "message": "Model berhasil diretraining dan disimpan!",
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "category": model_category
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------
# Routes
# ---------------------------

@login_manager.user_loader
def load_user(user_id):
    for user in users.values():
        if user.id == int(user_id):
            return user
    return None

USER_DATA = {
    "admin": "admin123"  
}


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in USER_DATA and USER_DATA[username] == password:
            session["user"] = username  
            flash("Login berhasil!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Username atau password salah!", "danger")

    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Berhasil logout!", "success")
    return redirect(url_for("home"))

@app.route('/home', methods=['GET'])
def home():
    create_interactive_plot()
    create_map(df_global)
    create_lineChart(df_global)
    barChart5(df_global)
    barChartLand(df_global)
    barChartFossil(df_global)
    status = growtStatus(df_global)


    return render_template('index.html', status=status, data=df_global.to_dict(orient="records"))

@app.route('/data')
def get_data():
    import pandas as pd
    df = pd.read_csv(LOCAL_FILE)
    df_cleaned = df.fillna(0) 
    return jsonify(df_cleaned.to_dict(orient="records"))


@app.route('/retrain-model', methods=['GET'])
def retrain_model():
    """Endpoint untuk retraining model."""
    result = train_lstm()
    return jsonify(result)

@app.route('/preprocessing', methods=['GET'])
def preprocessing():
    data = preprocess_data()
    return jsonify(data)

@app.route('/retrain')
def retrain():

    if "user" not in session:
        flash("Silakan login terlebih dahulu!", "warning")
        return redirect(url_for("login"))
    
    if not os.path.exists(LOCAL_FILE):
        return "Dataset belum tersedia. Silakan update data terlebih dahulu."

    df = pd.read_csv(LOCAL_FILE)
    df = df[df["Entity"] == "Indonesia"]
    df = df.dropna(subset=['Year', 'emissions_total_including_land_use_change'])
    df= df.tail(5)
    df = df.sort_values(by="Year", ascending=False)

    return render_template("retrain.html", data=df.to_dict(orient="records"), user=session["user"])

@app.route('/dashboard')
def dashboard():

    if "user" not in session: 
        flash("Silakan login terlebih dahulu!", "warning")
        return redirect(url_for("login"))
    
    if not os.path.exists(LOCAL_FILE):
        return "Dataset belum tersedia. Silakan update data terlebih dahulu."

    df = pd.read_csv(LOCAL_FILE)
    df = df[df["Entity"] == "Indonesia"]
    df = df.dropna(subset=['Year', 'emissions_total_including_land_use_change'])
    df = df.sort_values(by="Year", ascending=False)

    plt.figure(figsize=(10, 5))

   
    df_indonesia = df[df["Entity"] == "Indonesia"]
    if not df_indonesia.empty:
        plt.plot(df_indonesia["Year"], df_indonesia["emissions_total_including_land_use_change"], marker='o', linestyle='-', label="Total Emisi")
        plt.plot(df_indonesia["Year"], df_indonesia["emissions_from_land_use_change"], marker='o', linestyle='-', label="Land-use change")
        plt.plot(df_indonesia["Year"], df_indonesia["emissions_total"], marker='o', linestyle='-', label="Fossil Fuels")

    plt.xlabel("Tahun")
    plt.ylabel("Emisi CO2 (Juta Ton)")
    plt.title("Tren Emisi CO2 Indonesia")
    plt.legend()
    plt.grid(True)
    plt.savefig("static/co2_plot.png")
    plt.close()

    last_update = get_last_update()
    sorted_df = prediksi_df.sort_values(by='Year', ascending=False)


    table_html = sorted_df.to_html(classes='table table-striped', index=False, escape=False)


    table_html = table_html.replace('<th>', '<th style="text-align: left;">')

    return render_template("dashboard.html", data=df.to_dict(orient="records"), last_update=last_update, user=session["user"], tables=table_html)



@app.route('/update-data', methods=['GET'])
def update_data():
    """Endpoint untuk memperbarui dataset dari API."""
    result = update_dataset()
    return jsonify(result)

@app.route('/upload-data', methods=['POST'])
def upload_data():
    """Endpoint untuk mengunggah data manual dari file CSV/XLSX jika API gagal."""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "Tidak ada file yang diunggah."})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Nama file tidak valid."})

    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(file_path)  
            os.chmod(file_path, 0o644)  
            
           
            if filename.endswith('.csv'):
                df_uploaded = pd.read_csv(file_path, encoding='utf-8')
            else:
                df_uploaded = pd.read_excel(file_path)
            
            required_columns = {'Entity', 'Code', 'Year', 'emissions_total_including_land_use_change', 
                                'emissions_from_land_use_change', 'emissions_total'}
            missing_columns = required_columns - set(df_uploaded.columns)

            if missing_columns:
                return jsonify({"status": "error", "message": f"Kolom berikut tidak ditemukan: {missing_columns}"})

      
            if os.path.exists(LOCAL_FILE):
                os.chmod(LOCAL_FILE, 0o644) 
                df_lama = pd.read_csv(LOCAL_FILE, encoding='utf-8')
                df_updated = pd.concat([df_lama, df_uploaded]).drop_duplicates(subset=['Entity', 'Year']).reset_index(drop=True)
            else:
                df_updated = df_uploaded

            df_updated.to_csv(LOCAL_FILE, index=False, encoding='utf-8')
            save_update_log() 
            
            return jsonify({"status": "success", "message": "Data berhasil diunggah dan diperbarui!"})
        
        except PermissionError:
            return jsonify({"status": "error", "message": "Tidak memiliki izin membaca/mengedit file. Coba jalankan dengan sudo atau periksa izin file."})
        except Exception as e:
            return jsonify({"status": "error", "message": f"Kesalahan saat membaca file: {str(e)}"})

    else:
        return jsonify({"status": "error", "message": "Format file tidak didukung. Harap unggah CSV atau XLSX."})   

@app.route('/last-update', methods=['GET'])
def last_update():
    """Endpoint untuk mendapatkan waktu terakhir pembaruan data."""
    return jsonify({"last_update": get_last_update()})

if __name__ == '__main__':
    app.run(debug=True)

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     prediction = None
#     new_year = 0
#     new_emission = None

#     if request.method == 'POST':
#         try:
#             new_year = int(request.form['year'])
#             new_emission = float(request.form['emission'])
#             prediction = predict_next_year(model_user, new_year, new_emission)
#         except ValueError:
#             return render_template('predict.html', error="Masukkan angka yang valid!", prediction=None, new_year=0)

#     return render_template('predict.html', prediction=prediction, new_year=new_year)