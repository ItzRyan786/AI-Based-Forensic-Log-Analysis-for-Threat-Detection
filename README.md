# AI-Based-Forensic-Log-Analysis-for-Threat-Detection
Project

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import io
import json
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Import scikit-learn components
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False
    st.warning("scikit-learn not available. Some ML features will be limited.")

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except:
    AUTOREFRESH_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="AI-Based Behavioral Anomaly Detection for Insider Threats",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== SESSION STATE INITIALIZATION ==========
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'model': None,
        'scaler': None,
        'encoder': None,
        'trained': False,
        'running': False,
        'live_total': 0,
        'live_anomalies': 0,
        'anomaly_records': [],
        'rolling_logs': [],
        'threat_scores': {},
        'flash': True,
        'last_played': None,
        'live_counter': 0,
        'alerts': [],
        'investigation_cases': [],
        'data_source': 'synthetic',
        'real_data': None,
        'model_metrics': {},
        'user_profiles': {},
        'data_ingested': False,
        'df_processed': None,
        'training_features': [],
        'X_train': None,  # Added to persist training data for visualization
        'batch_alerts_generated': False,
        'batch_analysis_done': False,
        'report_data': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ========== DATA INGESTION MODULE ==========
class DataIngestion:
    """Handle multiple data sources and formats"""
    
    @staticmethod
    def parse_real_data(uploaded_file):
        """Parse uploaded real data files"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format")
                return None
            
            # Standardize column names
            column_mapping = {
                'user': 'User_ID',
                'user_id': 'User_ID',
                'username': 'User_ID',
                'employee_id': 'User_ID',
                'timestamp': 'Timestamp',
                'time': 'Timestamp',
                'date_time': 'Timestamp',
                'activity': 'Activity',
                'action': 'Activity',
                'event_type': 'Activity',
                'data_transferred': 'Volume_MB',
                'bytes': 'Volume_MB',
                'size_mb': 'Volume_MB',
                'volume': 'Volume_MB',
                'day': 'Day_of_Week',
                'weekday': 'Day_of_Week',
                'hour': 'Hour_of_Day',
                'time_hour': 'Hour_of_Day',
                'ip_address': 'IP_Address',
                'src_ip': 'IP_Address',
                'destination': 'Destination',
                'dest_ip': 'Destination',
                'severity': 'Severity',
                'risk_score': 'Risk_Score',
                'department': 'Department',
                'job_title': 'Job_Title',
                'location': 'Location'
            }
            
            df.columns = [column_mapping.get(col.lower().strip(), col) for col in df.columns]
            
            # Add missing required columns if needed
            required_columns = ['User_ID', 'Activity', 'Volume_MB']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'Volume_MB':
                        df[col] = np.random.exponential(10, len(df))
                    elif col == 'User_ID':
                        df[col] = [f'User_{i}' for i in range(1, len(df)+1)]
                    elif col == 'Activity':
                        df[col] = np.random.choice(['Login', 'File_Access', 'Email', 'Download'], len(df))
            
            # Parse timestamp if available
            if 'Timestamp' in df.columns:
                try:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                    df['Hour_of_Day'] = df['Timestamp'].dt.hour
                    df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
                    df['Date'] = df['Timestamp'].dt.date
                except:
                    pass
            
            # Add anomaly label if not present
            if 'True_Label' not in df.columns:
                df['True_Label'] = 0  # Assume normal initially
            
            return df
            
        except Exception as e:
            st.error(f"Error parsing file: {str(e)}")
            return None
    
    @staticmethod
    def generate_synthetic_data(n_normal=200, n_anomalies=40, include_features=True):
        """Generate enhanced synthetic data with more realistic patterns"""
        users = [f'EMP_{i:03d}' for i in range(1, 51)]  # 50 employees
        departments = ['IT', 'HR', 'Finance', 'Engineering', 'Sales', 'Marketing']
        job_titles = ['Manager', 'Developer', 'Analyst', 'Director', 'Intern']
        locations = ['NYC', 'SF', 'LON', 'TOK', 'DEL']
        
        data = []
        
        # Normal behavior patterns
        for _ in range(n_normal):
            u = np.random.choice(users)
            hour = int(np.random.normal(14, 3))
            hour = max(9, min(18, hour))  # Business hours
            day = np.random.randint(0, 5)  # Weekdays
            act = np.random.choice(['Login', 'File_Access', 'Email', 'Print', 'Meeting'], 
                                   p=[0.2, 0.4, 0.2, 0.1, 0.1])
            vol = np.random.exponential(5)
            
            record = {
                'User_ID': u,
                'Day_of_Week': day,
                'Hour_of_Day': hour,
                'Activity': act,
                'Volume_MB': vol,
                'True_Label': 0,
                'Department': np.random.choice(departments),
                'Job_Title': np.random.choice(job_titles),
                'Location': np.random.choice(locations),
                'IP_Address': f"10.0.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                'Timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 720))
            }
            
            # Add contextual features
            if include_features:
                record['Access_Frequency'] = np.random.poisson(5)
                record['Failed_Logins'] = np.random.poisson(0.1)
                record['Weekend_Activity'] = 0
                record['Unusual_Time'] = 0
            
            data.append(record)
        
        # Anomalous behavior patterns
        anomaly_patterns = [
            {
                'type': 'data_exfil', 
                'hour_ranges': [20, 21, 22, 23, 0, 1, 2, 3, 4],  # Overnight hours
                'activities': ['Cloud_Upload', 'USB_Transfer'], 
                'vol_multiplier': 10
            },
            {
                'type': 'privilege_abuse', 
                'hour_ranges': list(range(8, 18)),  # Business hours
                'activities': ['Sensitive_Access', 'Admin_Action'], 
                'vol_multiplier': 1
            },
            {
                'type': 'suspicious_timing', 
                'hour_ranges': list(range(0, 6)),  # Late night
                'activities': ['Login', 'File_Access'], 
                'vol_multiplier': 3
            },
            {
                'type': 'mass_download', 
                'hour_ranges': list(range(8, 18)),  # Business hours
                'activities': ['Bulk_Download'], 
                'vol_multiplier': 50
            },
        ]
        
        for _ in range(n_anomalies):
            u = np.random.choice(users)
            pattern_idx = np.random.choice(len(anomaly_patterns))
            pattern = anomaly_patterns[pattern_idx]
            
            # Select hour from available hour ranges
            hour = np.random.choice(pattern['hour_ranges'])
            
            # Set day based on pattern type
            if pattern['type'] == 'suspicious_timing':
                day = np.random.choice([5, 6])  # Weekend
            else:
                day = np.random.randint(0, 7)
            
            act = np.random.choice(pattern['activities'])
            vol = np.random.exponential(5) * pattern['vol_multiplier']
            
            record = {
                'User_ID': u,
                'Day_of_Week': day,
                'Hour_of_Day': hour,
                'Activity': act,
                'Volume_MB': vol,
                'True_Label': 1,
                'Department': np.random.choice(departments),
                'Job_Title': np.random.choice(job_titles),
                'Location': np.random.choice(locations),
                'IP_Address': f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                'Timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 24))
            }
            
            if include_features:
                record['Access_Frequency'] = np.random.poisson(20)  # Higher frequency
                record['Failed_Logins'] = np.random.poisson(3)  # More failed logins
                record['Weekend_Activity'] = 1 if day in [5, 6] else 0
                record['Unusual_Time'] = 1 if hour < 6 or hour > 20 else 0
            
            data.append(record)
        
        df = pd.DataFrame(data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        return df

# ========== THREAT INTELLIGENCE MODULE ==========
class ThreatIntelligence:
    """Threat scoring and risk assessment"""
    
    @staticmethod
    def calculate_risk_score(record, user_history=None):
        """Calculate comprehensive risk score"""
        base_score = 0
        
        # Activity-based scoring
        activity_weights = {
            'Cloud_Upload': 3.0,
            'USB_Transfer': 3.5,
            'Bulk_Download': 4.0,
            'Sensitive_Access': 3.0,
            'Admin_Action': 2.5,
            'File_Access': 1.0,
            'Login': 0.5,
            'Email': 0.3,
            'Print': 0.2,
            'USB_Connect': 3.0,
            'Meeting': 0.1
        }
        
        activity = record.get('Activity', '')
        base_score += activity_weights.get(activity, 1.0)
        
        # Volume-based scoring
        volume = record.get('Volume_MB', 0)
        if volume > 100:
            base_score += 2.0
        elif volume > 50:
            base_score += 1.0
        elif volume > 10:
            base_score += 0.5
        
        # Time-based scoring
        hour = record.get('Hour_of_Day', 12)
        if hour < 6 or hour > 20:  # Off-hours
            base_score += 2.0
        
        day = record.get('Day_of_Week', 0)
        if day >= 5:  # Weekend
            base_score += 1.5
        
        # User behavior context
        if user_history:
            user_avg_volume = user_history.get('avg_volume', 10)
            if volume > user_avg_volume * 3:
                base_score += 2.0
        
        # Additional features
        if record.get('Failed_Logins', 0) > 3:
            base_score += 2.0
        
        if record.get('Access_Frequency', 0) > 20:
            base_score += 1.5
        
        # Normalize score to 0-10
        risk_score = min(10.0, base_score)
        
        # Determine severity level
        if risk_score >= 8:
            severity = 'CRITICAL'
        elif risk_score >= 6:
            severity = 'HIGH'
        elif risk_score >= 4:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'
        
        return {
            'risk_score': round(risk_score, 2),
            'severity': severity,
            'factors': {
                'activity_weight': activity_weights.get(activity, 1.0),
                'volume_impact': 2.0 if volume > 100 else (1.0 if volume > 50 else 0.5 if volume > 10 else 0),
                'time_impact': 2.0 if (hour < 6 or hour > 20) else 0,
                'weekend_impact': 1.5 if day >= 5 else 0
            }
        }
    
    @staticmethod
    def generate_alert(record, risk_assessment):
        """Generate structured alert"""
        alert_id = hashlib.md5(f"{record.get('User_ID')}_{datetime.now().timestamp()}".encode()).hexdigest()[:8]
        
        alert = {
            'alert_id': alert_id,
            'timestamp': datetime.now(),
            'user_id': record.get('User_ID'),
            'activity': record.get('Activity'),
            'volume_mb': record.get('Volume_MB'),
            'risk_score': risk_assessment['risk_score'],
            'severity': risk_assessment['severity'],
            'factors': risk_assessment['factors'],
            'status': 'NEW',
            'assigned_to': None,
            'investigation_notes': []
        }
        
        return alert
    
    @staticmethod
    def batch_analyze_data(df, threshold=6.0):
        """Analyze entire dataset and generate alerts"""
        alerts = []
        threat_intel = ThreatIntelligence()
        
        for idx, row in df.iterrows():
            record = row.to_dict()
            risk_assessment = threat_intel.calculate_risk_score(record)
            
            if risk_assessment['risk_score'] >= threshold:
                alert = threat_intel.generate_alert(record, risk_assessment)
                alerts.append(alert)
        
        return alerts

# ========== MACHINE LEARNING MODULE ==========
class MLModel:
    """Enhanced machine learning models for threat detection"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
    
    def prepare_features(self, df, feature_columns=None):
        """Prepare features for model training"""
        df_processed = df.copy()
        
        if feature_columns is None:
            # Default feature columns
            feature_columns = [
                'Day_of_Week', 'Hour_of_Day', 'Volume_MB',
                'Access_Frequency', 'Failed_Logins', 
                'Weekend_Activity', 'Unusual_Time'
            ]
        
        # Filter to columns that exist
        existing_features = [f for f in feature_columns if f in df_processed.columns]
        
        # Encode categorical variables
        if 'Activity' in df_processed.columns:
            le = LabelEncoder()
            df_processed['Activity_Encoded'] = le.fit_transform(df_processed['Activity'])
            self.encoders['activity'] = le
            if 'Activity_Encoded' not in existing_features:
                existing_features.append('Activity_Encoded')
        
        # Handle missing values
        for col in existing_features:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            else:
                # Add missing columns with default values
                df_processed[col] = 0
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(df_processed[existing_features])
        self.scalers['main'] = scaler
        
        return X, df_processed, existing_features
    
    def train_isolation_forest(self, X, contamination=0.05, n_estimators=100):
        """Train Isolation Forest model"""
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X)
        self.models['isolation_forest'] = model
        return model
    
    def train_random_forest(self, X, y, n_estimators=100):
        """Train Random Forest classifier"""
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        model.fit(X, y)
        self.models['random_forest'] = model
        return model
    
    def predict(self, model_type, X):
        """Make predictions using specified model"""
        if model_type in self.models:
            model = self.models[model_type]
            
            if model_type == 'isolation_forest':
                preds = model.predict(X)
                # Convert -1 (anomaly) to 1, 1 (normal) to 0
                preds = (preds == -1).astype(int)
                probs = model.decision_function(X)
                # Normalize to 0-1
                if len(np.unique(probs)) > 1:
                    probs = 1 - (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)
                else:
                    probs = np.ones_like(probs) * 0.5
            else:
                preds = model.predict(X)
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(X)
                        # Check if predict_proba returns a 2D array with multiple columns
                        if proba.ndim == 2 and proba.shape[1] > 1:
                            probs = proba[:, 1]  # Probability of positive class (anomaly)
                        else:
                            # If only one class is present or 1D array is returned
                            probs = proba.ravel() if proba.ndim == 2 else proba
                    except:
                        # Fallback to using predict method
                        probs = preds.astype(float)
                else:
                    probs = preds.astype(float)
            
            return preds, probs
        
        return None, None

# ========== VISUALIZATION MODULE ==========
class Visualization:
    """Enhanced visualization components"""
    
    @staticmethod
    def create_threat_timeline(alerts):
        """Create timeline visualization of threats"""
        if not alerts:
            fig = go.Figure()
            fig.add_annotation(text="No alerts to display", showarrow=False)
            fig.update_layout(height=300)
            return fig
        
        df = pd.DataFrame(alerts)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        fig = px.scatter(
            df, x='timestamp', y='risk_score',
            color='severity', size='volume_mb',
            hover_data=['user_id', 'activity', 'severity'],
            title="Threat Alert Timeline"
        )
        
        fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
        fig.update_layout(height=400)
        
        return fig
    
    @staticmethod
    def create_user_risk_heatmap(df):
        """Create heatmap of user risk by hour"""
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data to display", showarrow=False)
            fig.update_layout(height=300)
            return fig
        
        # Calculate risk scores if not present
        if 'Risk_Score' not in df.columns:
            threat_intel = ThreatIntelligence()
            risk_scores = []
            for _, row in df.iterrows():
                risk = threat_intel.calculate_risk_score(row.to_dict())
                risk_scores.append(risk['risk_score'])
            df['Risk_Score'] = risk_scores
        
        heatmap_data = df.pivot_table(
            index='User_ID', 
            columns='Hour_of_Day', 
            values='Risk_Score',
            aggfunc='mean',
            fill_value=0
        )
        
        fig = px.imshow(
            heatmap_data,
            title="User Risk Heatmap (by Hour)",
            color_continuous_scale='RdYlGn_r',
            labels=dict(x="Hour of Day", y="User", color="Risk Score")
        )
        
        fig.update_layout(height=500)
        return fig
    
    @staticmethod
    def create_behavior_clusters(df, features):
        """Create 3D visualization of behavior clusters"""
        if len(df) < 10:
            return None
        
        try:
            pca = PCA(n_components=3)
            components = pca.fit_transform(features)
            
            df_vis = df.copy()
            df_vis['PCA1'] = components[:, 0]
            df_vis['PCA2'] = components[:, 1]
            df_vis['PCA3'] = components[:, 2]
            
            fig = px.scatter_3d(
                df_vis, x='PCA1', y='PCA2', z='PCA3',
                color='True_Label' if 'True_Label' in df_vis.columns else 'Volume_MB',
                hover_data=['User_ID', 'Activity', 'Volume_MB'],
                title="Behavior Pattern Clusters (3D PCA)",
                opacity=0.7
            )
            
            fig.update_layout(height=600)
            return fig
        except:
            return None

# ========== REPORTING MODULE ==========
class Reporting:
    """Enhanced reporting functionality"""

    @staticmethod
    def generate_comprehensive_report(alerts, df=None, report_type="daily"):
        """Generate comprehensive report with statistics"""
        
        # Initialize empty structure if inputs are missing
        if alerts is None:
            alerts = []
            
        report = {
            "report_id": f"RPT_{int(time.time())}",
            "generated_at": datetime.now().isoformat(),
            "report_type": report_type,
            "summary": {
                "total_alerts": len(alerts),
                "generated_by": "AI Insider Threat Detection System"
            },
            "alerts": {},
            "statistics": {},
            "recommendations": []
        }

        # Process alerts if available
        if alerts:
            alerts_df = pd.DataFrame(alerts)
            
            if 'severity' in alerts_df.columns:
                # Convert to standard python ints for JSON serialization
                severity_counts = alerts_df['severity'].value_counts().to_dict()
                report["alerts"]["by_severity"] = {k: int(v) for k, v in severity_counts.items()}
            
            if 'user_id' in alerts_df.columns:
                top_users = alerts_df['user_id'].value_counts().head(5).to_dict()
                report["alerts"]["top_users"] = {k: int(v) for k, v in top_users.items()}
            
            if 'timestamp' in alerts_df.columns:
                # Handle timestamp conversion safely
                try:
                    timestamps = pd.to_datetime(alerts_df['timestamp'])
                    report["alerts"]["time_range"] = {
                        "first": timestamps.min().isoformat(),
                        "last": timestamps.max().isoformat()
                    }
                except:
                    report["alerts"]["time_range"] = "Time data unavailable"

        # Process dataset statistics if available
        if df is not None and not df.empty:
            report["statistics"]["total_records"] = int(len(df))
            report["statistics"]["total_users"] = int(df['User_ID'].nunique()) if 'User_ID' in df.columns else 0
            
            if 'Volume_MB' in df.columns:
                # Explicit float conversion for JSON compatibility
                report["statistics"]["volume_stats"] = {
                    "mean": float(df['Volume_MB'].mean()),
                    "max": float(df['Volume_MB'].max()),
                    "min": float(df['Volume_MB'].min()),
                    "std": float(df['Volume_MB'].std())
                }
            
            if 'Activity' in df.columns:
                activity_dist = df['Activity'].value_counts().head(10).to_dict()
                report["statistics"]["activity_distribution"] = {k: int(v) for k, v in activity_dist.items()}

        # Add recommendations logic
        high_severity = report.get("alerts", {}).get("by_severity", {}).get("HIGH", 0)
        critical_severity = report.get("alerts", {}).get("by_severity", {}).get("CRITICAL", 0)
        
        if critical_severity > 0:
            report["recommendations"].append(f"Immediate investigation required for {critical_severity} critical alerts.")
        
        if high_severity > 5:
            report["recommendations"].append("Consider enhancing monitoring rules for high-frequency alerts.")
            
        if not alerts and (df is None or df.empty):
            report["recommendations"].append("System is running but no data or alerts were provided for this report period.")
        
        return report

    @staticmethod
    def export_report(report, format="json"):
        """Export report in different formats"""
        if format == "json":
            # Custom encoder function to handle potential NumPy types
            def np_encoder(object):
                if isinstance(object, (np.generic, np.integer, np.floating)):
                    return object.item()
                if isinstance(object, (datetime, pd.Timestamp)):
                    return object.isoformat()
                return str(object)
                
            return json.dumps(report, indent=2, default=np_encoder)
            
        elif format == "csv":
            # Flatten dictionary for CSV (Simplified)
            if "summary" in report:
                flat_data = []
                # Add summary stats
                flat_data.append({"Category": "Summary", "Metric": "Total Alerts", "Value": report["summary"].get("total_alerts")})
                
                # Add alert stats
                if "by_severity" in report.get("alerts", {}):
                    for sev, count in report["alerts"]["by_severity"].items():
                        flat_data.append({"Category": "Alert Severity", "Metric": sev, "Value": count})
                
                # Add volume stats
                if "volume_stats" in report.get("statistics", {}):
                    for stat, val in report["statistics"]["volume_stats"].items():
                        flat_data.append({"Category": "Volume Stats", "Metric": stat, "Value": val})
                        
                return pd.DataFrame(flat_data).to_csv(index=False)
            return "No report data to convert to CSV"
        return ""

# ========== MAIN APPLICATION ==========
def main():
    init_session_state()
    
    # Sidebar Navigation
    st.sidebar.title("Insider Threat Detection")
    st.sidebar.markdown("---")
    
    menu = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Data Ingestion", "Model Training", 
         "Live Detection", "Alerts & Cases", "Analytics & Reporting"]
    )
    
    # Initialize modules
    data_ingestor = DataIngestion()
    threat_intel = ThreatIntelligence()
    reporting = Reporting()
    
    # ========== DASHBOARD PAGE ==========
    if menu == "Dashboard":
        st.title("AI-Based Behavioral Anomaly Detection for Insider Threats")
        st.markdown("""
        ### Security Operations Center
        Monitor, detect, and respond to insider threats in real-time.
        """)
        
        # Display current data source status
        if st.session_state.data_source == 'real':
            st.success(f"Dashboard Source: Real Data Uploaded ({len(st.session_state.real_data) if st.session_state.real_data is not None else 0} records)")
        else:
            st.info("Dashboard Source: Synthetic Data Generation")
        
        # Get data based on source
        if st.session_state.data_source == 'real' and st.session_state.real_data is not None:
            df = st.session_state.real_data
        else:
            df = data_ingestor.generate_synthetic_data(n_normal=500, n_anomalies=50)
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", df['User_ID'].nunique())
        
        with col2:
            st.metric("Total Activities", len(df))
        
        with col3:
            anomalies = df[df['True_Label'] == 1] if 'True_Label' in df.columns else pd.DataFrame()
            st.metric("Potential Threats", len(anomalies))
        
        with col4:
            avg_risk = df['Volume_MB'].mean() if len(df) > 0 else 0
            st.metric("Avg Data Volume", f"{avg_risk:.1f} MB")
        
        # Visualizations
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Activity distribution
            if 'Activity' in df.columns:
                activity_counts = df['Activity'].value_counts().reset_index()
                activity_counts.columns = ['Activity', 'Count']
                fig1 = px.pie(
                    activity_counts, values='Count', names='Activity',
                    title="Activity Distribution",
                    hole=0.3
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            # Time-based analysis
            if 'Hour_of_Day' in df.columns:
                hourly_counts = df.groupby('Hour_of_Day').size().reset_index(name='count')
                fig2 = px.bar(
                    hourly_counts, x='Hour_of_Day', y='count',
                    title="Activity by Hour of Day",
                    labels={'Hour_of_Day': 'Hour', 'count': 'Activity Count'}
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        with col_right:
            # Volume distribution
            if 'Volume_MB' in df.columns:
                fig3 = px.histogram(
                    df, x='Volume_MB',
                    title="Data Volume Distribution",
                    nbins=50,
                    log_y=True
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            # User activity heatmap
            viz = Visualization()
            heatmap_fig = viz.create_user_risk_heatmap(df)
            st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # ========== DATA INGESTION PAGE ==========
    elif menu == "Data Ingestion":
        st.title("Data Ingestion & Preparation")
        
        # --- NEW CENTRALIZED DATA SELECTION ---
        st.markdown("### 1. Select Data Source")
        
        # Default index calculation
        default_idx = 1 if st.session_state.data_source == 'real' else 0
        
        source_selection = st.radio(
            "Choose the source of data for analysis:",
            ["Synthetic Data (Auto-Generated)", "Real Data (File Upload)"],
            index=default_idx,
            horizontal=True
        )
        
        # Update session state based on selection
        if source_selection == "Real Data (File Upload)":
            st.session_state.data_source = 'real'
        else:
            st.session_state.data_source = 'synthetic'
            
        st.markdown("---")
        
        # --- TABS FOR WORKFLOW ---
        if st.session_state.data_source == 'real':
            tab1, tab2, tab3 = st.tabs(["Upload Data", "Data Preview", "Feature Engineering"])
            
            with tab1:
                st.subheader("Upload Security Logs")
                st.info("Upload your security logs here. Supported formats: CSV, JSON, Parquet, Excel.")
                
                uploaded_files = st.file_uploader(
                    "Upload security log files",
                    type=['csv', 'json', 'parquet', 'xlsx'],
                    accept_multiple_files=True
                )
                
                if uploaded_files:
                    all_data = []
                    for file in uploaded_files:
                        df_file = data_ingestor.parse_real_data(file)
                        if df_file is not None:
                            all_data.append(df_file)
                    
                    if all_data:
                        df_combined = pd.concat(all_data, ignore_index=True)
                        st.session_state.real_data = df_combined
                        st.session_state.data_ingested = True
                        st.success(f"Successfully processed {len(df_combined)} records from {len(uploaded_files)} files.")
                    
            with tab2:
                st.subheader("Data Preview & Statistics")
                if st.session_state.real_data is not None:
                    df = st.session_state.real_data
                    st.dataframe(df.head(100), use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Data Statistics")
                        stats = {
                            "Total Records": len(df),
                            "Total Users": df['User_ID'].nunique(),
                        }
                        if 'Timestamp' in df.columns:
                            stats["Date Range"] = {"Start": str(df['Timestamp'].min()), "End": str(df['Timestamp'].max())}
                        st.json(stats)
                    
                    with col2:
                        st.subheader("Column Info")
                        col_info = pd.DataFrame({
                            'Column': df.columns,
                            'Missing %': [f"{(df[col].isnull().sum() / len(df) * 100):.1f}%" for col in df.columns]
                        })
                        st.dataframe(col_info, use_container_width=True)
                    
                    # Batch Analysis Button
                    st.markdown("---")
                    col_an1, col_an2 = st.columns([1, 3])
                    with col_an1:
                         threshold = st.slider("Risk Threshold", 0.0, 10.0, 6.0, 0.5)
                    with col_an2:
                        st.write("")
                        st.write("")
                        if st.button("Run Batch Analysis on Uploaded Data", type="primary"):
                            with st.spinner("Analyzing..."):
                                batch_alerts = threat_intel.batch_analyze_data(df, threshold)
                                if batch_alerts:
                                    st.session_state.alerts.extend(batch_alerts)
                                    report = reporting.generate_comprehensive_report(batch_alerts, df, "batch_analysis")
                                    st.session_state.report_data = report
                                    st.success(f"Generated {len(batch_alerts)} alerts!")
                else:
                    st.warning("Please upload files in the 'Upload Data' tab first.")
                    
            with tab3:
                st.subheader("Feature Engineering")
                if st.session_state.real_data is not None:
                    df = st.session_state.real_data.copy()
                    st.info("System will extract: Hour of Day, Weekend Flags, and Off-Hour Flags.")
                    
                    if 'Timestamp' in df.columns:
                        df['Hour_of_Day'] = pd.to_datetime(df['Timestamp']).dt.hour
                        df['Day_of_Week'] = pd.to_datetime(df['Timestamp']).dt.dayofweek
                        df['Is_Weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)
                        df['Is_Off_Hours'] = df['Hour_of_Day'].apply(lambda x: 1 if x < 6 or x > 20 else 0)
                    
                    st.dataframe(df.head(), use_container_width=True)
                    
                    if st.button("Apply Engineered Features"):
                        st.session_state.real_data = df
                        st.success("Features applied to dataset.")
                else:
                    st.warning("Please upload files first.")

        # --- VIEW FOR SYNTHETIC DATA ---
        else:
            st.info("You are currently using Synthetic Data. The system will auto-generate logs for demonstration.")
            st.subheader("Synthetic Data Preview")
            
            df_synth = data_ingestor.generate_synthetic_data(n_normal=100, n_anomalies=10)
            st.dataframe(df_synth.head(50), use_container_width=True)
            
            st.markdown("To use your own data, select 'Real Data (File Upload)' above.")

    # ========== MODEL TRAINING PAGE ==========
    elif menu == "Model Training":
        st.title("Machine Learning Model Training")
        
        if not SKLEARN_AVAILABLE:
            st.error("scikit-learn is not installed. Please install it with: pip install scikit-learn")
            st.stop()
        
        # Get data based on the centralized selection
        if st.session_state.data_source == 'real' and st.session_state.real_data is not None:
            df = st.session_state.real_data
            st.success("Training on: Uploaded Real Data")
        else:
            df = data_ingestor.generate_synthetic_data(n_normal=500, n_anomalies=50)
            if st.session_state.data_source == 'real':
                st.warning("Real data selected but none uploaded. Falling back to synthetic data for training.")
            else:
                st.info("Training on: Synthetic Data")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Model Configuration")
            
            model_type = st.selectbox(
                "Select Model Algorithm",
                ["Isolation Forest", "Random Forest"]
            )
            
            if model_type == "Isolation Forest":
                contamination = st.slider("Contamination Rate", 0.01, 0.3, 0.05, 0.01)
                n_estimators = st.slider("Number of Trees", 50, 500, 100)
            
            elif model_type == "Random Forest":
                n_estimators = st.slider("Number of Trees", 50, 500, 100)
                max_depth = st.slider("Max Depth", 5, 50, 20)
            
            test_size = st.slider("Test Set Size (%)", 10, 40, 20)
        
        with col2:
            st.subheader("Feature Selection")
            
            # Available features
            available_features = [
                'Day_of_Week', 'Hour_of_Day', 'Volume_MB',
                'Access_Frequency', 'Failed_Logins', 
                'Weekend_Activity', 'Unusual_Time',
                'Is_Weekend', 'Is_Off_Hours',
                'Avg_Volume', 'Std_Volume', 'Max_Volume', 'Activity_Count'
            ]
            
            # Filter to features that exist in data
            existing_features = [f for f in available_features if f in df.columns]
            
            selected_features = st.multiselect(
                "Select features for training:",
                existing_features,
                default=existing_features[:min(5, len(existing_features))]
            )
        
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                # Initialize ML model
                ml_model = MLModel()
                
                # Prepare features
                X, df_processed, features = ml_model.prepare_features(df, selected_features)
                
                # Save X for visualization
                st.session_state.X_train = X

                # Get labels
                y = df_processed['True_Label'].values if 'True_Label' in df_processed.columns else np.zeros(len(df_processed))
                
                # Train model based on type
                if model_type == "Isolation Forest":
                    model = ml_model.train_isolation_forest(X, contamination, n_estimators)
                    predictions, probabilities = ml_model.predict('isolation_forest', X)
                    
                    if len(np.unique(y)) > 1:
                        # Calculate metrics
                        accuracy = accuracy_score(y, predictions)
                        precision = precision_score(y, predictions, zero_division=0)
                        recall = recall_score(y, predictions, zero_division=0)
                        f1 = f1_score(y, predictions, zero_division=0)
                        
                        st.session_state.model_metrics = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'model_type': 'Isolation Forest'
                        }
                    else:
                        st.session_state.model_metrics = {
                            'model_type': 'Isolation Forest',
                            'note': 'No true labels available for evaluation'
                        }
                
                elif model_type == "Random Forest":
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=42
                    )
                    
                    model = ml_model.train_random_forest(X_train, y_train, n_estimators)
                    predictions, probabilities = ml_model.predict('random_forest', X_test)
                    
                    # Calculate metrics
                    unique_classes = np.unique(y_test)
                    if len(unique_classes) > 1:
                        accuracy = accuracy_score(y_test, predictions)
                        precision = precision_score(y_test, predictions, zero_division=0)
                        recall = recall_score(y_test, predictions, zero_division=0)
                        f1 = f1_score(y_test, predictions, zero_division=0)
                        
                        try:
                            auc = roc_auc_score(y_test, probabilities)
                        except:
                            auc = 0.5
                        
                        st.session_state.model_metrics = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'auc': auc,
                            'model_type': 'Random Forest'
                        }
                    else:
                        st.session_state.model_metrics = {
                            'model_type': 'Random Forest',
                            'note': f'Only one class ({unique_classes[0] if len(unique_classes) > 0 else "unknown"}) present in test set'
                        }
                
                # Store in session state
                st.session_state.model = ml_model
                st.session_state.trained = True
                st.session_state.training_features = features
                st.session_state.df_processed = df_processed
                
                st.success("Model trained successfully!")
                
                # Show metrics
                if st.session_state.model_metrics:
                    st.subheader("Model Performance Metrics")
                    
                    if 'note' in st.session_state.model_metrics:
                        st.info(st.session_state.model_metrics['note'])
                    else:
                        metrics = st.session_state.model_metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                        with col2:
                            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
                        with col3:
                            st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
                        with col4:
                            st.metric("F1-Score", f"{metrics.get('f1', 0):.3f}")
                        
                        if 'auc' in metrics:
                            st.metric("AUC-ROC", f"{metrics['auc']:.3f}")
        
        # Visualization of results
        if st.session_state.trained and st.session_state.df_processed is not None:
            st.subheader("Model Results Visualization")
            
            tab1, tab2 = st.tabs(["PCA Projection", "Feature Importance"])
            
            with tab1:
                # Check for X_train availability (Fixed error: 'X' not associated with a value)
                if st.session_state.X_train is not None:
                    try:
                        df_vis = st.session_state.df_processed.copy()
                        X_vis = st.session_state.X_train

                        if st.session_state.model and 'isolation_forest' in st.session_state.model.models:
                            # Get predictions
                            predictions, _ = st.session_state.model.predict('isolation_forest', X_vis)
                            df_vis['Prediction'] = predictions
                        
                        # Use PCA for visualization
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_vis)
                        
                        df_vis['PCA1'] = X_pca[:, 0]
                        df_vis['PCA2'] = X_pca[:, 1]
                        
                        color_col = 'Prediction' if 'Prediction' in df_vis.columns else 'True_Label'
                        
                        fig = px.scatter(
                            df_vis, x='PCA1', y='PCA2',
                            color=color_col,
                            hover_data=['User_ID', 'Activity', 'Volume_MB'],
                            title="PCA Projection of User Activities",
                            opacity=0.7
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create PCA visualization: {str(e)}")
                else:
                    st.warning("Please train the model to generate PCA visualization.")
            
            with tab2:
                st.subheader("Feature Importance Analysis")
                
                if st.session_state.trained and st.session_state.model is not None:
                    ml_model = st.session_state.model
                    
                    # Check if we have a Random Forest model with feature importance
                    if 'random_forest' in ml_model.models:
                        rf_model = ml_model.models['random_forest']
                        features = st.session_state.training_features
                        
                        # Get feature importances
                        importances = rf_model.feature_importances_
                        
                        # Create DataFrame for visualization
                        importance_df = pd.DataFrame({
                            'Feature': features,
                            'Importance': importances
                        }).sort_values('Importance', ascending=True)
                        
                        # Plot horizontal bar chart
                        fig = px.bar(
                            importance_df, 
                            x='Importance', 
                            y='Feature',
                            orientation='h',
                            title="Random Forest Feature Importance",
                            color='Importance',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(
                            height=max(400, len(features) * 25),
                            xaxis_title="Feature Importance Score",
                            yaxis_title="Feature"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display as table with additional metrics
                        st.subheader("Detailed Feature Analysis")
                        
                        # Calculate permutation importance if possible
                        try:
                            from sklearn.inspection import permutation_importance
                            
                            # Get data for permutation importance
                            if st.session_state.data_source == 'real' and st.session_state.real_data is not None:
                                df_perm = st.session_state.real_data
                            else:
                                df_perm = data_ingestor.generate_synthetic_data(n_normal=100, n_anomalies=10)
                            
                            X_perm, df_processed_perm, features_perm = ml_model.prepare_features(df_perm, features)
                            y_perm = df_processed_perm['True_Label'].values if 'True_Label' in df_processed_perm.columns else np.zeros(len(df_processed_perm))
                            
                            # Split data
                            X_train_perm, X_test_perm, y_train_perm, y_test_perm = train_test_split(
                                X_perm, y_perm, test_size=0.3, random_state=42
                            )
                            
                            # Calculate permutation importance
                            result = permutation_importance(
                                rf_model, X_test_perm, y_test_perm,
                                n_repeats=10,
                                random_state=42,
                                n_jobs=-1
                            )
                            
                            # Add permutation importance to DataFrame
                            importance_df['Permutation_Importance'] = result.importances_mean
                            importance_df['Permutation_Std'] = result.importances_std
                            
                        except ImportError:
                            st.info("Permutation importance requires scikit-learn >= 0.24.0")
                        
                        # Display the importance table
                        importance_df = importance_df.sort_values('Importance', ascending=False)
                        st.dataframe(
                            importance_df,
                            use_container_width=True,
                            column_config={
                                'Feature': st.column_config.TextColumn('Feature'),
                                'Importance': st.column_config.NumberColumn(
                                    'Importance',
                                    format="%.4f",
                                    help="Higher values indicate more important features"
                                ),
                                'Permutation_Importance': st.column_config.NumberColumn(
                                    'Permutation Importance',
                                    format="%.4f",
                                    help="Mean decrease in accuracy when feature is shuffled"
                                ),
                                'Permutation_Std': st.column_config.NumberColumn(
                                    'Permutation Std',
                                    format="%.4f",
                                    help="Standard deviation of permutation importance"
                                )
                            }
                        )
                        
                        # Feature importance insights
                        st.subheader("Feature Importance Insights")
                        
                        top_features = importance_df.head(5)
                        bottom_features = importance_df.tail(5)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Top 5 Most Important Features:**")
                            for idx, row in top_features.iterrows():
                                st.markdown(f"• **{row['Feature']}**: {row['Importance']:.4f}")
                        
                        with col2:
                            st.markdown("**Bottom 5 Least Important Features:**")
                            for idx, row in bottom_features.iterrows():
                                st.markdown(f"• **{row['Feature']}**: {row['Importance']:.4f}")
                        
                        # Recommendation based on feature importance
                        st.markdown("**Recommendations:**")
                        if importance_df['Importance'].max() / importance_df['Importance'].min() > 100:
                            st.info("Large difference in feature importance detected. Consider feature selection to remove low-importance features.")
                        else:
                            st.info("Feature importances are relatively balanced. All features contribute to the model.")
                    
                    elif 'isolation_forest' in ml_model.models:
                        # Fixed: Model-Agnostic Feature Importance (Perturbation Analysis)
                        # This replaces the brittle 'decision_path' check
                        st.info("Calculating feature importance using model-agnostic perturbation analysis...")
                        
                        iso_model = ml_model.models['isolation_forest']
                        
                        if st.session_state.X_train is not None:
                            X_eval = st.session_state.X_train.copy()
                            features = st.session_state.training_features
                            
                            # Optimization for large datasets to prevent UI freezing
                            if len(X_eval) > 1000:
                                indices = np.random.choice(len(X_eval), 1000, replace=False)
                                X_eval = X_eval[indices]
                            
                            try:
                                # 1. Get baseline anomaly scores (lower is more anomalous usually, but decision_function output varies)
                                # We care about magnitude of change
                                baseline_scores = iso_model.decision_function(X_eval)
                                
                                importances = []
                                for i in range(X_eval.shape[1]):
                                    # 2. Perturb one feature by shuffling it
                                    X_temp = X_eval.copy()
                                    np.random.shuffle(X_temp[:, i])
                                    
                                    # 3. Measure change in anomaly scores
                                    perturbed_scores = iso_model.decision_function(X_temp)
                                    
                                    # Importance = Mean Absolute Deviation from baseline
                                    # If shuffling a feature changes scores significantly, it is important
                                    imp = np.mean(np.abs(baseline_scores - perturbed_scores))
                                    importances.append(imp)
                                
                                # Normalize importances to 0-1 range for plotting
                                importances = np.array(importances)
                                if importances.sum() > 0:
                                    importances = importances / importances.sum()
                                
                                # Create visualization
                                importance_df = pd.DataFrame({
                                    'Feature': features,
                                    'Importance': importances
                                }).sort_values('Importance', ascending=True)
                                
                                fig = px.bar(
                                    importance_df, 
                                    x='Importance', 
                                    y='Feature',
                                    orientation='h',
                                    title="Isolation Forest Feature Importance (Perturbation Analysis)",
                                    color='Importance',
                                    color_continuous_scale='Blues'
                                )
                                fig.update_layout(
                                    height=max(300, len(features) * 20),
                                    xaxis_title="Relative Importance (Impact on Anomaly Score)",
                                    yaxis_title="Feature"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.success("Feature importance calculated successfully using perturbation analysis.")
                                
                            except Exception as e:
                                st.warning(f"Could not calculate feature importance: {str(e)}")
                        else:
                            st.warning("Please train the model to view feature importance.")
                    else:
                        st.warning("No model available for feature importance analysis.")
                else:
                    st.info("Please train a model first to see feature importance analysis.")
    
    # ========== LIVE DETECTION PAGE ==========
    elif menu == "Live Detection":
        st.title("Real-time Threat Detection")
        
        if not st.session_state.trained:
            st.warning("Please train a model first in the 'Model Training' section")
            st.stop()
        
        # Control panel
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Controls")
            
            if st.button("Start Live Monitoring", type="primary"):
                st.session_state.running = True
                st.rerun()
            
            if st.button("Pause Monitoring"):
                st.session_state.running = False
                st.rerun()
            
            if st.button("Reset Counters"):
                st.session_state.live_total = 0
                st.session_state.live_anomalies = 0
                st.session_state.anomaly_records = []
                st.session_state.rolling_logs = []
                st.session_state.alerts = []
                st.rerun()
            
            st.markdown("---")
            st.subheader("Settings")
            
            alert_threshold = st.slider("Alert Threshold", 0.0, 10.0, 6.0, 0.5)
            refresh_rate = st.slider("Refresh Rate (ms)", 1000, 10000, 2000)
        
        with col1:
            # Live metrics
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric("Live Events", st.session_state.live_total)
            
            with metrics_col2:
                st.metric("Active Threats", st.session_state.live_anomalies)
            
            with metrics_col3:
                detection_rate = (st.session_state.live_anomalies / max(1, st.session_state.live_total)) * 100
                st.metric("Detection Rate", f"{detection_rate:.1f}%")
            
            with metrics_col4:
                avg_response = 0.5  # Simulated response time
                st.metric("Avg Response", f"{avg_response}s")
        
        # Main display area
        if st.session_state.running:
            if AUTOREFRESH_AVAILABLE:
                st_autorefresh(interval=refresh_rate, key="live_monitoring")
            
            # Generate simulated live data
            threat_intel = ThreatIntelligence()
            
            # Create new event
            if st.session_state.data_source == 'real' and st.session_state.real_data is not None:
                # Sample from real data
                df_real = st.session_state.real_data
                sample_idx = np.random.randint(0, len(df_real))
                event = df_real.iloc[sample_idx].to_dict()
            else:
                # Generate synthetic event
                df_synth = data_ingestor.generate_synthetic_data(n_normal=1, n_anomalies=0)
                event = df_synth.iloc[0].to_dict()
                # Occasionally add anomaly
                if np.random.random() < 0.3:  # 30% chance of anomaly
                    df_anomaly = data_ingestor.generate_synthetic_data(n_normal=0, n_anomalies=1)
                    event = df_anomaly.iloc[0].to_dict()
            
            # Prepare event for prediction
            ml_model = st.session_state.model
            features = st.session_state.training_features
            
            # Encode activity if needed
            if 'Activity' in event and 'activity' in ml_model.encoders:
                try:
                    event['Activity_Encoded'] = ml_model.encoders['activity'].transform([event['Activity']])[0]
                except:
                    event['Activity_Encoded'] = 0
            
            # Create feature vector
            feature_vector = []
            for feat in features:
                if feat in event:
                    feature_vector.append(event[feat])
                else:
                    feature_vector.append(0)  # Default value
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Scale features
            if 'main' in ml_model.scalers:
                feature_vector = ml_model.scalers['main'].transform(feature_vector)
            
            # Make prediction
            if 'isolation_forest' in ml_model.models:
                prediction, confidence = ml_model.predict('isolation_forest', feature_vector)
                event['Is_Anomaly'] = prediction[0]
                event['Confidence'] = confidence[0]
            elif 'random_forest' in ml_model.models:
                prediction, confidence = ml_model.predict('random_forest', feature_vector)
                event['Is_Anomaly'] = prediction[0]
                event['Confidence'] = confidence[0]
            else:
                event['Is_Anomaly'] = 0
                event['Confidence'] = 0
            
            # Calculate risk score
            risk_assessment = threat_intel.calculate_risk_score(event)
            event['Risk_Score'] = risk_assessment['risk_score']
            event['Severity'] = risk_assessment['severity']
            event['Timestamp'] = datetime.now()
            
            # Update counters
            st.session_state.live_total += 1
            if event['Is_Anomaly'] or event['Risk_Score'] >= alert_threshold:
                st.session_state.live_anomalies += 1
                st.session_state.anomaly_records.append(event)
                
                # Generate alert
                alert = threat_intel.generate_alert(event, risk_assessment)
                st.session_state.alerts.append(alert)
            
            # Add to rolling logs
            st.session_state.rolling_logs.append(event)
            if len(st.session_state.rolling_logs) > 50:
                st.session_state.rolling_logs.pop(0)
            
            # Display current alert
            if event['Is_Anomaly'] or event['Risk_Score'] >= alert_threshold:
                severity_color = {
                    'CRITICAL': '#ff4444',
                    'HIGH': '#ff8800',
                    'MEDIUM': '#ffbb33',
                    'LOW': '#00C851'
                }
                
                color = severity_color.get(event['Severity'], '#ff4444')
                
                alert_html = f"""
                <div style="border-left: 5px solid {color}; padding: 15px; background-color: rgba(255, 68, 68, 0.1); border-radius: 5px; margin: 10px 0;">
                    <h4 style="margin: 0; color: {color};">
                        {event['Severity']} ALERT - {event['User_ID']}
                    </h4>
                    <p style="margin: 5px 0;">
                        <b>Activity:</b> {event['Activity']} | 
                        <b>Volume:</b> {event['Volume_MB']:.1f} MB | 
                        <b>Risk Score:</b> {event['Risk_Score']:.1f}
                    </p>
                    <p style="margin: 5px 0; font-size: 0.9em; color: #666;">
                        Time: {datetime.now().strftime('%H:%M:%S')} | 
                        Confidence: {(event['Confidence']*100):.1f}%
                    </p>
                </div>
                """
                st.markdown(alert_html, unsafe_allow_html=True)
            else:
                st.info("Normal activity detected")
            
            # Display recent logs
            st.subheader("Recent Events")
            if st.session_state.rolling_logs:
                recent_df = pd.DataFrame(st.session_state.rolling_logs[-10:])
                display_cols = ['User_ID', 'Activity', 'Volume_MB', 'Risk_Score', 'Severity']
                display_cols = [col for col in display_cols if col in recent_df.columns]
                st.dataframe(
                    recent_df[display_cols],
                    use_container_width=True,
                    hide_index=True
                )
            
            # Visualizations
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                if len(st.session_state.rolling_logs) > 5:
                    trend_df = pd.DataFrame(st.session_state.rolling_logs)
                    trend_df['Index'] = range(len(trend_df))
                    
                    if 'Risk_Score' in trend_df.columns:
                        fig_trend = px.line(
                            trend_df, x='Index', y='Risk_Score',
                            title="Risk Score Trend",
                            markers=True
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)
            
            with col_viz2:
                if st.session_state.anomaly_records:
                    alerts_df = pd.DataFrame(st.session_state.anomaly_records)
                    if 'Severity' in alerts_df.columns:
                        severity_counts = alerts_df['Severity'].value_counts().reset_index()
                        severity_counts.columns = ['Severity', 'Count']
                        
                        fig_severity = px.pie(
                            severity_counts, values='Count', names='Severity',
                            title="Alert Severity Distribution",
                            color='Severity',
                            color_discrete_map={
                                'CRITICAL': '#ff4444',
                                'HIGH': '#ff8800',
                                'MEDIUM': '#ffbb33',
                                'LOW': '#00C851'
                            }
                        )
                        st.plotly_chart(fig_severity, use_container_width=True)
        else:
            st.info("Live monitoring is paused. Click 'Start Live Monitoring' to begin.")
            
            # Show historical data if available
            if st.session_state.anomaly_records:
                st.subheader("Historical Anomalies")
                historical_df = pd.DataFrame(st.session_state.anomaly_records)
                display_cols = ['User_ID', 'Activity', 'Volume_MB', 'Risk_Score', 'Severity']
                display_cols = [col for col in display_cols if col in historical_df.columns]
                st.dataframe(
                    historical_df[display_cols],
                    use_container_width=True
                )
    
    # ========== ALERTS & CASES PAGE ==========
    elif menu == "Alerts & Cases":
        st.title("Alerts Management & Investigation")
        
        tab1, tab2, tab3 = st.tabs(["Active Alerts", "Investigation Cases", "Alert Analytics"])
        
        with tab1:
            st.subheader("Active Alerts")
            
            if st.session_state.alerts:
                alerts_df = pd.DataFrame(st.session_state.alerts)
                
                # Filter options
                col_filter1, col_filter2, col_filter3 = st.columns(3)
                
                with col_filter1:
                    severity_filter = st.multiselect(
                        "Filter by Severity",
                        ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
                        default=['CRITICAL', 'HIGH']
                    )
                
                with col_filter2:
                    status_filter = st.multiselect(
                        "Filter by Status",
                        ['NEW', 'INVESTIGATING', 'RESOLVED', 'FALSE_POSITIVE'],
                        default=['NEW', 'INVESTIGATING']
                    )
                
                with col_filter3:
                    if st.button("Refresh Alerts"):
                        st.rerun()
                
                # Apply filters
                if 'severity' in alerts_df.columns and 'status' in alerts_df.columns:
                    filtered_alerts = alerts_df[
                        (alerts_df['severity'].isin(severity_filter)) &
                        (alerts_df['status'].isin(status_filter))
                    ]
                else:
                    filtered_alerts = alerts_df
                
                # Display alerts
                for idx, alert in filtered_alerts.iterrows():
                    severity_color = {
                        'CRITICAL': '#ff4444',
                        'HIGH': '#ff8800',
                        'MEDIUM': '#ffbb33',
                        'LOW': '#00C851'
                    }
                    
                    color = severity_color.get(alert.get('severity', 'MEDIUM'), '#ff4444')
                    
                    with st.expander(f"Alert {alert.get('alert_id', idx)} - {alert.get('severity', 'UNKNOWN')} - {alert.get('user_id', 'Unknown')}"):
                        col_alert1, col_alert2 = st.columns([2, 1])
                        
                        with col_alert1:
                            st.markdown(f"**Activity:** {alert.get('activity', 'Unknown')}")
                            st.markdown(f"**Volume:** {alert.get('volume_mb', 0):.1f} MB")
                            st.markdown(f"**Risk Score:** {alert.get('risk_score', 0):.2f}")
                            st.markdown(f"**Timestamp:** {alert.get('timestamp', 'Unknown')}")
                        
                        with col_alert2:
                            # Alert actions
                            current_status = alert.get('status', 'NEW')
                            new_status = st.selectbox(
                                "Update Status",
                                ['NEW', 'INVESTIGATING', 'RESOLVED', 'FALSE_POSITIVE'],
                                key=f"status_{idx}",
                                index=['NEW', 'INVESTIGATING', 'RESOLVED', 'FALSE_POSITIVE'].index(current_status) if current_status in ['NEW', 'INVESTIGATING', 'RESOLVED', 'FALSE_POSITIVE'] else 0
                            )
                            
                            if st.button("Update", key=f"update_{idx}"):
                                # Update alert status
                                st.session_state.alerts[idx]['status'] = new_status
                                st.success("Status updated!")
                                st.rerun()
                            
                            if st.button("Create Case", key=f"case_{idx}"):
                                # Create investigation case
                                case = {
                                    'case_id': f"CASE_{len(st.session_state.investigation_cases)+1:04d}",
                                    'alert_id': alert.get('alert_id', idx),
                                    'user_id': alert.get('user_id', 'Unknown'),
                                    'created_at': datetime.now(),
                                    'status': 'OPEN',
                                    'severity': alert.get('severity', 'MEDIUM'),
                                    'investigator': None,
                                    'notes': [],
                                    'evidence': []
                                }
                                st.session_state.investigation_cases.append(case)
                                st.success("Investigation case created!")
                            
                            if st.button("Ignore", key=f"ignore_{idx}"):
                                st.session_state.alerts[idx]['status'] = 'FALSE_POSITIVE'
                                st.warning("Alert marked as false positive")
                                st.rerun()
            else:
                st.info("No active alerts to display.")
        
        with tab2:
            st.subheader("Investigation Cases")
            
            if st.session_state.investigation_cases:
                cases_df = pd.DataFrame(st.session_state.investigation_cases)
                
                for idx, case in cases_df.iterrows():
                    with st.expander(f"Case {case.get('case_id', idx)} - {case.get('severity', 'UNKNOWN')} - {case.get('user_id', 'Unknown')}"):
                        col_case1, col_case2 = st.columns([2, 1])
                        
                        with col_case1:
                            st.markdown(f"**Status:** {case.get('status', 'OPEN')}")
                            st.markdown(f"**Created:** {case.get('created_at', 'Unknown')}")
                            st.markdown(f"**Alert ID:** {case.get('alert_id', 'Unknown')}")
                            
                            # Notes section
                            st.markdown("### Investigation Notes")
                            new_note = st.text_area("Add note", key=f"note_{idx}")
                            
                            if st.button("Add Note", key=f"add_note_{idx}"):
                                if new_note:
                                    note_entry = {
                                        'timestamp': datetime.now(),
                                        'investigator': 'Analyst',
                                        'note': new_note
                                    }
                                    st.session_state.investigation_cases[idx]['notes'].append(note_entry)
                                    st.success("Note added!")
                                    st.rerun()
                        
                        with col_case2:
                            # Case actions
                            current_status = case.get('status', 'OPEN')
                            new_status = st.selectbox(
                                "Case Status",
                                ['OPEN', 'IN_PROGRESS', 'RESOLVED', 'ESCALATED'],
                                key=f"case_status_{idx}",
                                index=['OPEN', 'IN_PROGRESS', 'RESOLVED', 'ESCALATED'].index(current_status) if current_status in ['OPEN', 'IN_PROGRESS', 'RESOLVED', 'ESCALATED'] else 0
                            )
                            
                            investigator = st.text_input("Investigator", value=case.get('investigator', ''), key=f"investigator_{idx}")
                            
                            if st.button("Update Case", key=f"update_case_{idx}"):
                                st.session_state.investigation_cases[idx]['status'] = new_status
                                st.session_state.investigation_cases[idx]['investigator'] = investigator
                                st.success("Case updated!")
                                st.rerun()
            else:
                st.info("No investigation cases yet.")
        
        with tab3:
            st.subheader("Alert Analytics")
            
            if st.session_state.alerts:
                alerts_df = pd.DataFrame(st.session_state.alerts)
                
                col_analytics1, col_analytics2 = st.columns(2)
                
                with col_analytics1:
                    # Alert trends over time
                    if 'timestamp' in alerts_df.columns:
                        alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
                        alerts_df['date'] = alerts_df['timestamp'].dt.date
                        
                        daily_alerts = alerts_df.groupby('date').size().reset_index(name='count')
                        
                        if len(daily_alerts) > 1:
                            fig_daily = px.line(
                                daily_alerts, x='date', y='count',
                                title="Daily Alert Trends",
                                markers=True
                            )
                            st.plotly_chart(fig_daily, use_container_width=True)
                
                with col_analytics2:
                    # Alert status distribution
                    if 'status' in alerts_df.columns:
                        status_counts = alerts_df['status'].value_counts().reset_index()
                        status_counts.columns = ['Status', 'Count']
                        
                        if len(status_counts) > 0:
                            fig_status = px.pie(
                                status_counts, values='Count', names='Status',
                                title="Alert Status Distribution"
                            )
                            st.plotly_chart(fig_status, use_container_width=True)
            else:
                st.info("No alert data available for analytics.")
    
    # ========== ANALYTICS & REPORTING PAGE ==========
    elif menu == "Analytics & Reporting":
        st.title("Advanced Analytics & Reporting")
        
        tab1, tab2, tab3 = st.tabs(["Behavior Analytics", "Risk Assessment", "Reporting & Export"])
        
        # Ensure we have data to analyze (Real or Synthetic)
        if st.session_state.data_source == 'real' and st.session_state.real_data is not None:
            analysis_df = st.session_state.real_data
        else:
            # Generate synthetic data if no real data is loaded
            analysis_df = data_ingestor.generate_synthetic_data(n_normal=500, n_anomalies=50)

        with tab1:
            st.subheader("User Behavior Analytics")
            
            if not analysis_df.empty and 'User_ID' in analysis_df.columns and 'Activity' in analysis_df.columns:
                # User behavior profiles
                user_behavior = analysis_df.groupby('User_ID').agg({
                    'Activity': 'count'
                }).round(2)
                
                user_behavior.columns = ['Activity_Count']
                
                # Add volume metrics if available
                if 'Volume_MB' in analysis_df.columns:
                    volume_stats = analysis_df.groupby('User_ID')['Volume_MB'].agg(['sum', 'mean', 'max']).round(2)
                    user_behavior = pd.concat([user_behavior, volume_stats], axis=1)
                    user_behavior.columns = ['Activity_Count', 'Total_Volume', 'Avg_Volume', 'Max_Volume']
                
                st.dataframe(user_behavior.head(20), use_container_width=True)
                
                # Add Chart
                st.subheader("Top Users by Data Volume")
                top_volume_users = user_behavior.sort_values('Total_Volume', ascending=False).head(10)
                st.bar_chart(top_volume_users['Total_Volume'])
            else:
                st.info("No data available for behavior analytics.")

        with tab2:
            st.subheader("Risk Assessment Dashboard")
            
            # Check if we have anomalies/alerts to analyze
            if st.session_state.anomaly_records or st.session_state.alerts:
                # Combine historical anomalies and alerts
                records_to_use = st.session_state.anomaly_records + st.session_state.alerts
                risk_df = pd.DataFrame(records_to_use)
                
                # Normalize columns if needed
                if 'Risk_Score' in risk_df.columns:
                    pass 
                elif 'risk_score' in risk_df.columns:
                    risk_df.rename(columns={'risk_score': 'Risk_Score', 'user_id': 'User_ID'}, inplace=True)

                if 'User_ID' in risk_df.columns and 'Risk_Score' in risk_df.columns:
                    # Aggregate risk by user
                    user_risk = risk_df.groupby('User_ID').agg({
                        'Risk_Score': ['mean', 'max', 'sum']
                    }).round(2)
                    
                    # Flatten multi-level columns
                    user_risk.columns = ['Avg_Risk', 'Max_Risk', 'Total_Risk']
                    
                    # Sort by total risk
                    user_risk = user_risk.sort_values('Total_Risk', ascending=False)
                    
                    st.dataframe(user_risk, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Risk Distribution")
                        fig = px.histogram(risk_df, x="Risk_Score", nbins=20, title="Risk Score Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Data format issue: Missing User_ID or Risk_Score in records.")
            else:
                st.info("No risk assessments generated yet. Run 'Batch Analysis' or 'Live Detection' to generate risk data.")

        with tab3:
            st.subheader("Reporting & Export")
            
            col_report1, col_report2 = st.columns(2)
            
            with col_report1:
                st.markdown("### Generate Reports")
                
                report_type = st.selectbox(
                    "Report Type",
                    ["Daily Summary", "Weekly Analysis", "Monthly Report", "Custom Period", "Batch Analysis"]
                )
                
                include_charts = st.checkbox("Include Charts", value=True)
                export_format = st.selectbox("Export Format", ["JSON", "CSV"])
                
                # Generate report button
                if st.button("Generate Report", type="primary"):
                    with st.spinner("Generating comprehensive report..."):
                        
                        # Generate report using the fixed Reporting class
                        report = reporting.generate_comprehensive_report(
                            st.session_state.alerts, 
                            analysis_df, # Pass the guaranteed DF (real or synthetic)
                            report_type
                        )
                        
                        st.session_state.report_data = report
                        
                        st.success("Report generated successfully!")
                        
                        # Display report summary
                        st.markdown("### Report Summary")
                        st.json(report.get("summary", {}))
                        
                        # Display alert statistics
                        if report.get("alerts"):
                            st.markdown("### Alert Statistics")
                            col_stat1, col_stat2, col_stat3 = st.columns(3)
                            
                            with col_stat1:
                                total_alerts = report.get("summary", {}).get("total_alerts", 0)
                                st.metric("Total Alerts", total_alerts)
                            
                            with col_stat2:
                                if report["alerts"].get("by_severity"):
                                    critical = report["alerts"]["by_severity"].get("CRITICAL", 0)
                                    st.metric("Critical Alerts", critical)
                            
                            with col_stat3:
                                if report["alerts"].get("by_severity"):
                                    high = report["alerts"]["by_severity"].get("HIGH", 0)
                                    st.metric("High Alerts", high)

            with col_report2:
                st.markdown("### Report Preview & Export")
                
                if st.session_state.report_data:
                    report = st.session_state.report_data
                    
                    # Display key metrics
                    st.markdown("#### Key Metrics")
                    
                    metrics_cols = st.columns(3)
                    with metrics_cols[0]:
                        if "summary" in report and "total_alerts" in report["summary"]:
                            st.metric("Total Alerts", report["summary"]["total_alerts"])
                    
                    with metrics_cols[1]:
                        if "statistics" in report and "total_records" in report["statistics"]:
                            st.metric("Total Records", report["statistics"]["total_records"])
                    
                    with metrics_cols[2]:
                        if "statistics" in report and "total_users" in report["statistics"]:
                            st.metric("Unique Users", report["statistics"]["total_users"])
                    
                    # Export options
                    st.markdown("#### Export Options")
                    
                    if export_format == "JSON":
                        json_report = reporting.export_report(report, "json")
                        st.download_button(
                            label="Download JSON Report",
                            data=json_report,
                            file_name=f"threat_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    elif export_format == "CSV":
                        # Generate CSV data
                        csv_data = reporting.export_report(report, "csv")
                        
                        st.download_button(
                            label="Download CSV Summary",
                            data=csv_data,
                            file_name=f"threat_report_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Separate download for full alerts if they exist
                        if st.session_state.alerts:
                            alerts_csv = pd.DataFrame(st.session_state.alerts).to_csv(index=False)
                            st.download_button(
                                label="Download Full Alerts (CSV)",
                                data=alerts_csv,
                                file_name=f"full_alerts_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                    # Display recommendations if available
                    if report.get("recommendations"):
                        st.markdown("#### Recommendations")
                        for i, rec in enumerate(report["recommendations"], 1):
                            st.markdown(f"{i}. {rec}")
                
                else:
                    st.info("No report generated yet. Click 'Generate Report' to create one.")

# Run the application
if __name__ == "__main__":
    main()
