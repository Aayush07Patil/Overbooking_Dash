import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from flask import request, jsonify
import os

# Try to import pyodbc, but provide alternative if it fails
try:
    import pyodbc
except ImportError:
    print("pyodbc not installed. Using sample data only.")
    pyodbc = None

# Initialize the Dash app
app = dash.Dash(__name__, 
                title="Overbooking Dashboard", 
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Expose Flask server to add custom routes

# Global variables to store the last received data
current_flight_data = {
    "flight_no": "",
    "flight_date": datetime.now().date().isoformat(),
    "flight_origin": "",
    "flight_destination": ""
}

# Layout - simplified with fixed height/width
app.layout = html.Div([
    # Graph with loading indicator
    dcc.Loading(
        id="loading-graph",
        type="circle",
        color="#119DFF",
        children=[
            html.Div(id="graph-container", style={"height": "100vh", "width": "100%"})
        ]
    ),
    
    # Hidden div to store the flight data from the .NET application
    html.Div(id="flight-data-store", style={"display": "none"}),
    
    # Add interval component to trigger updates
    dcc.Interval(
        id='interval-component',
        interval=1200000,  # in milliseconds (20 minutes)
        n_intervals=0
    )
], style={"height": "100vh", "width": "100%", "margin": "0px", "padding": "0px", "overflow": "hidden"})

# Function to connect to the database and get data
def get_flight_data(flight_no, flight_date, origin, destination):
    # For demonstration purposes, if DB connection fails, use sample data
    try:
        if not pyodbc:
            raise ImportError("pyodbc is not available")

        # Get database connection details from environment variables
        db_server = os.environ.get('DB_SERVER', '')
        db_name = os.environ.get('DB_NAME', '')
        db_user = os.environ.get('DB_USER', '')
        db_password = os.environ.get('DB_PASSWORD', '')
        
        # Check if we have all the required connection details
        if not all([db_server, db_name, db_user, db_password]):
            print("Missing database connection details. Using sample data instead...")
            raise Exception("Missing database connection details")
        
        # Try connecting to the database
        conn_str = (
            f'DRIVER={{ODBC Driver 17 for SQL Server}};'
            f'SERVER={db_server};'
            f'DATABASE={db_name};'
            f'UID={db_user};'
            f'PWD={db_password};'
            f'Encrypt=yes;'
            f'TrustServerCertificate=no;'
            f'Connection Timeout=30;'
        )
        
        conn = pyodbc.connect(conn_str)
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
        if isinstance(flight_date, str):
            try:
                flight_date = datetime.strptime(flight_date.split('T')[0], "%Y-%m-%d").date()
            except:
                flight_date = datetime.now().date()
                
        sample_dates = [flight_date + timedelta(days=i) for i in range(15)]
        
        # Create sample data - ensure everything is a proper type
        sample_data = {
            'FltNo': [flight_no] * 15,
            'FltDate': sample_dates,
            'Origin': [origin] * 15,
            'Destination': [destination] * 15,
            'ReportWeight': [float(1000 + i * 50 + (100 * (i % 3))) for i in range(15)],  # Ensure float
            'ReportVolume': [float(500 + i * 25 + (50 * (i % 4))) for i in range(15)],    # Ensure float
            'OBW': [float(i * 2.5 - 10) for i in range(15)]  # Ensure float
        }
        
        df = pd.DataFrame(sample_data)
        print(f"Generated sample data: {len(df)} rows")
        return df

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

# New API endpoint to reset data
@server.route('/reset-data', methods=['POST'])
def reset_data():
    global current_flight_data
    
    try:
        # Reset the current flight data to empty values
        current_flight_data = {
            "flight_no": "",
            "flight_date": datetime.now().date().isoformat(),
            "flight_origin": "",
            "flight_destination": ""
        }
        
        print("Dashboard data reset successfully")
        
        return jsonify({"status": "success", "message": "Data reset successfully"}), 200
    
    except Exception as e:
        print(f"Error resetting data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

# Callback to update the graph container based on stored flight data
@callback(
    Output("graph-container", "children"),
    [Input("interval-component", "n_intervals")]
)
def update_graphs(n_intervals):
    # Get the current flight data
    flight_no = current_flight_data["flight_no"]
    flight_date = current_flight_data["flight_date"]
    origin = current_flight_data["flight_origin"]
    destination = current_flight_data["flight_destination"]
    
    if not all([flight_no, flight_date, origin, destination]):
        # Return empty message if missing data
        return html.Div("Waiting for flight data...", 
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                            "height": "100%",
                            "fontSize": "16px"
                        })
    
    # Get data from database
    df = get_flight_data(flight_no, flight_date, origin, destination)
    
    if df.empty:
        return html.Div("No data found for the given parameters.", 
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                            "height": "100%",
                            "fontSize": "16px"
                        })
    
    # Convert FltDate to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['FltDate']):
        df['FltDate'] = pd.to_datetime(df['FltDate'], errors='coerce')
    
    # Format dates for display
    df['FormattedDate'] = df['FltDate'].dt.strftime('%d %b').str.upper()
    
    # Ensure we have the OBW column
    if 'OBW' not in df.columns:
        df['OBW'] = 0  # Default overbooking values
    
    # Convert columns to numeric to ensure they're plotted correctly
    for col in ['ReportWeight', 'ReportVolume', 'OBW']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Calculate max OBW value and set y-axis max to 1000 higher
    max_obw = df['OBW'].max()
    y_axis_max = 1000 + max_obw
    
    # Create figure with weight graph and OBW values
    fig = go.Figure()
    
    # Add weight bars
    fig.add_trace(
        go.Bar(
            x=df['FormattedDate'],
            y=df['ReportWeight'],
            name='Weight (kg)',
            marker_color='lightblue'
        )
    )
    
    # Add OBW line
    fig.add_trace(
        go.Scatter(
            x=df['FormattedDate'],
            y=df['OBW'],
            name='OBW',
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=8, symbol='circle')
        )
    )
    
    # Update layout with single Y-axis and title in plot
    fig.update_layout(
        title=dict(
            text="Weight and Overbooking Status",
            x=0.5,  # Center title
            y=0.95  # Position near top
        ),
        xaxis_title="Flight Date",
        yaxis=dict(
            title="Weight (kg) / OBW",
            rangemode="tozero",  # Make y-axis start from zero
            range=[0, y_axis_max]  # Set range to 1000 higher than max OBW
        ),
        legend=dict(
            x=1.05,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        template='plotly_white',
        margin=dict(l=50, r=100, t=50, b=50),  # Increased top margin for title
        height=700,
        hovermode="x unified"
    )
    
    # Create the graph component with explicit dimensions
    graph = dcc.Graph(
        id='weight-obw-graph',
        figure=fig,
        style={'height': '100%', 'width': '100%'},
        config={'responsive': True, 'displayModeBar': False}
    )
    
    return graph

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))