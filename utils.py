import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sqlalchemy.orm import Session
from database import SatelliteData, GroundMeasurement

def load_satellite_data(file_path):
    """Load and preprocess satellite data."""
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
            return data, transform, crs
    except Exception as e:
        return None, None, None

def save_satellite_data(db: Session, data, transform, timestamp=None):
    """Save satellite data to database."""
    if timestamp is None:
        timestamp = datetime.utcnow()

    rows, cols = data.shape
    resolution = transform[0]  # pixel size in coordinate system units

    for i in range(rows):
        for j in range(cols):
            if not np.isnan(data[i, j]):
                lon, lat = transform * (j, i)
                db_entry = SatelliteData(
                    timestamp=timestamp,
                    latitude=lat,
                    longitude=lon,
                    no2_value=float(data[i, j]),
                    resolution=resolution,
                    source='satellite'
                )
                db.add(db_entry)

    db.commit()

def load_ground_data(file_path):
    """Load ground station measurement data."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        return None

def save_ground_measurements(db: Session, data):
    """Save ground measurements to database."""
    for _, row in data.iterrows():
        measurement = GroundMeasurement(
            timestamp=datetime.utcnow(),
            latitude=row['latitude'],
            longitude=row['longitude'],
            no2_value=row['no2_value'],
            station_name=row.get('station_name', 'unknown')
        )
        db.add(measurement)
    db.commit()

def create_no2_map(data, title="NO2 Concentration Map"):
    """Create an interactive map visualization using plotly."""
    fig = px.imshow(
        data,
        labels=dict(color="NO2 (μg/m³)"),
        title=title,
        color_continuous_scale="RdYlBu_r"
    )
    fig.update_layout(
        title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

def calculate_metrics(y_true, y_pred):
    """Calculate model performance metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

def handle_missing_data(data, method='interpolate'):
    """Handle missing or invalid data points."""
    if method == 'interpolate':
        return pd.DataFrame(data).interpolate(method='linear').values
    else:
        return np.nan_to_num(data, nan=np.nanmean(data))