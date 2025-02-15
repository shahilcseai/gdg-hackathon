import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NO2DownscalingModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, data):
        """Prepare features for the model."""
        rows, cols = data.shape
        X = []
        y = []
        
        # Create spatial features
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(data[i, j]):
                    X.append([
                        i/rows,  # normalized row position
                        j/cols,  # normalized column position
                        data[i, j]  # NO2 value
                    ])
                    y.append(data[i, j])
        
        return np.array(X), np.array(y)
    
    def train(self, data):
        """Train the downscaling model."""
        X, y = self.prepare_features(data)
        X = self.scaler.fit_transform(X)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        return X_val, y_val
    
    def predict(self, data, scale_factor=2):
        """Generate high-resolution predictions."""
        rows, cols = data.shape
        new_rows = rows * scale_factor
        new_cols = cols * scale_factor
        
        # Create high-resolution grid
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, new_cols),
            np.linspace(0, 1, new_rows)
        )
        
        X_pred = np.column_stack([
            grid_x.ravel(),
            grid_y.ravel(),
            np.repeat(data.ravel(), scale_factor**2)
        ])
        
        X_pred = self.scaler.transform(X_pred)
        predictions = self.model.predict(X_pred)
        
        return predictions.reshape(new_rows, new_cols)
