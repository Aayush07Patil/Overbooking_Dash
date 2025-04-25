import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from flask import request, jsonify

# Try to import pyodbc, but provide alternative if it fails
try:
    import pyodbc
except ImportError:
    print("pyodbc not installed. Using sample data only.")
    pyodbc = None

# Initialize the Dash app
app = dash.Dash(__name__, title="Overbooking Dashboard", suppress_callback_exceptions=True)
server = app.server  # Expose Flask server to add custom routes

# Global variables to store the last received data
current_flight_data = {
    "flight_no": "",
    "flight_date": datetime.now().date().isoformat(),
    "flight_origin": "",
    "flight_destination": ""
}

# Create the layout - removed input forms, added display panel
app.layout = html.Div([
    html.Div([
        dcc.Loading(
            id="loading-graphs",
            type="circle",
            children=[
                html.Div([
                    html.H3("Weight and Overbooking Status", style={"textAlign": "center"}),
                    dcc.Graph(id="weight-graph")
                ], style={"width": "100%", "display": "inline-block"})
            ]
        )
    ], style={"marginTop": "20px"}),
    
    # Hidden div to store the flight data from the .NET application
    html.Div(id="flight-data-store", style={"display": "none"}),
    
    # Add interval component to trigger updates
    dcc.Interval(
        id='interval-component',
        interval=300000,  # in milliseconds (30 seconds)
        n_intervals=0
    )
])

# Function to connect to the database and get data
def get_flight_data(flight_no, flight_date, origin, destination):
    # For demonstration purposes, if DB connection fails, use sample data
    try:
        # For Azure SQL Database, Windows Authentication (Trusted_Connection) won't work
        # We need to use SQL Authentication with the proper driver
        try:
            # Method 1: Using ODBC Driver 17 (recommended for Azure SQL)
            conn = pyodbc.connect(
                'DRIVER={ODBC Driver 17 for SQL Server};'
                'SERVER=qidtestingindia.database.windows.net;'  # Remove the port from server name
                'DATABASE=rm-demo-erp-db;'
                'UID=rmdemodeploymentuser;'  # Replace with actual username
                'PWD=rm#demo#2515;'  # Replace with actual password
                'Encrypt=yes;'  # Required for Azure SQL
                'TrustServerCertificate=no;'
                'Connection Timeout=30;'
            )
        except Exception as e1:
            print(f"First connection attempt failed: {e1}")
            try:
                # Method 2: Using SQL Server driver as fallback
                conn = pyodbc.connect(
                    'DRIVER={SQL Server};'
                    'SERVER=qidtestingindia.database.windows.net;'  # Remove the port from server name
                    'DATABASE=rm-demo-erp-db;'
                    'UID=rmdemodeploymentuser;'  # Replace with actual username
                    'PWD=rm#demo#2515;'  # Replace with actual password
                    'Encrypt=yes;'  # Required for Azure SQL
                )
            except Exception as e2:
                print(f"Second connection attempt failed: {e2}")
                # If both connection methods fail, raise exception to use sample data
                raise Exception("Cannot connect to database")
        
        # Create a cursor
        cursor = conn.cursor()
        
        # Format the date for SQL query
        # Convert flight_date to datetime if it's a string
        if isinstance(flight_date, str):
            try:
                flight_date = datetime.strptime(flight_date.split('T')[0], "%Y-%m-%d").date()
            except:
                flight_date = datetime.now().date()
                
        formatted_date = flight_date.strftime("%Y-%m-%d")
        
        # Construct and execute the query to get the next 15 instances
        query = """
        SELECT TOP 15 FltNo, FltDate, Origin, Destination, ReportWeight, ReportVolume, OBW 
        FROM dbo.CapacityTransaction
        WHERE FltNo = ?
        AND Origin = ?
        AND Destination = ?
        AND FltDate >= ?
        ORDER BY FltDate
        """
        
        cursor.execute(query, (flight_no, origin, destination, formatted_date))
        
        # Fetch all results
        rows = cursor.fetchall()
        
        # Create DataFrame from results
        columns = [column[0] for column in cursor.description]
        df = pd.DataFrame.from_records(rows, columns=columns)
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        return df
        
    except Exception as e:
        print(f"Database error: {e}")
        print("Using sample data instead...")
        
        # Generate sample data for demonstration
        sample_dates = [flight_date + timedelta(days=i) for i in range(15)]
        
        # Create sample data
        sample_data = {
            'FltNo': [flight_no] * 15,
            'FltDate': sample_dates,
            'Origin': [origin] * 15,
            'Destination': [destination] * 15,
            'ReportWeight': [round(1000 + i * 50 + (100 * (i % 3)), 2) for i in range(15)],  # Random-ish weights
            'ReportVolume': [round(500 + i * 25 + (50 * (i % 4)), 2) for i in range(15)],     # Random-ish volumes
            'OBW': [round(i * 2.5 - 10, 1) for i in range(15)]  # Sample overbooking values, some negative, some positive
        }
        
        return pd.DataFrame(sample_data)

# API endpoint to receive data from .NET application
@server.route('/update-data', methods=['POST'])
def update_data():
    global current_flight_data
    
    try:
        # Get the data from the request
        data = request.get_json()
        
        # Update the current flight data
        current_flight_data = {
            "flight_no": data.get("flight_no", ""),
            "flight_date": data.get("flight_date", datetime.now().date().isoformat()),
            "flight_origin": data.get("flight_origin", ""),
            "flight_destination": data.get("flight_destination", "")
        }
        
        print(f"Received data: {current_flight_data}")
        
        return jsonify({"status": "success", "message": "Data received successfully"}), 200
    
    except Exception as e:
        print(f"Error processing data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

# Callback to update the graph based on stored flight data
@callback(
    Output("weight-graph", "figure"),
    [Input("interval-component", "n_intervals")]
)
def update_graphs(n_intervals):
    # Get the current flight data
    flight_no = current_flight_data["flight_no"]
    flight_date = current_flight_data["flight_date"]
    origin = current_flight_data["flight_origin"]
    destination = current_flight_data["flight_destination"]
    
    if not all([flight_no, flight_date, origin, destination]):
        # Return empty figure if no data is available
        empty_fig = go.Figure()
        empty_fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{
                "text": "Waiting for flight data...",
                "showarrow": False,
                "font": {"size": 16}
            }]
        )
        
        return empty_fig
    
    # Get data from database
    df = get_flight_data(flight_no, flight_date, origin, destination)
    
    if df.empty:
        # Return empty figure if no data
        empty_fig = go.Figure()
        empty_fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{
                "text": "No data found for the given parameters.",
                "showarrow": False,
                "font": {"size": 16}
            }]
        )
        
        return empty_fig
    
    # Convert FltDate to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['FltDate']):
        df['FltDate'] = pd.to_datetime(df['FltDate'])
    
    # Ensure we have the OBW column
    if 'OBW' not in df.columns:
        df['OBW'] = 0  # Default overbooking values
    
    # Convert OBW to numeric (if it's a string)
    df['OBW'] = pd.to_numeric(df['OBW'], errors='coerce').fillna(0)
    
    # Create figure with weight graph and OBW values
    fig = go.Figure()
    
    # Add weight trace
    fig.add_trace(
        go.Bar(
            x=df['FltDate'],
            y=df['ReportWeight'],
            name='Weight',
            marker_color='lightblue'
        )
    )
    
    # Add OBW line chart as an overlay
    fig.add_trace(
        go.Scatter(
            x=df['FltDate'],
            y=df['OBW'],
            name='Overbooking (OBW)',
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=8, symbol='circle')
        )
    )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(
            title="Flight Date",
            tickformat="%d %b",  # Format the date as "DD MMM" (e.g., "07 JUN")
            tickmode="array",    # Force all ticks to be shown
            tickvals=df['FltDate'],  # Show ticks for all dates in the dataset
            ticktext=[date.strftime("%d %b").upper() for date in df['FltDate']]  # Format date labels
        ),
        yaxis=dict(
            title="Weight (kg)",
            rangemode="tozero"  # Make y-axis start from zero
        ),
        legend=dict(x=0.02, y=0.98),
        hovermode="x unified",
        # Add annotations for OBW values - safely handle string conversion
        annotations=[
            dict(
                x=date,
                y=obw_val,
                text=f"{obw_val:.1f}" if obw_val > 0 else f"{obw_val:.1f}" if obw_val < 0 else "0",
                showarrow=False,
                yshift=10,
                font=dict(
                    color="black",
                    size=10
                )
            )
            for date, obw_val in zip(df['FltDate'], df['OBW'])
        ]
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0',port=int(os.environ.get('PORT',8050)))