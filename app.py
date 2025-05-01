import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta, date
from flask import request, jsonify
import os
import numpy as np

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
                
        # Calculate the start date (15 days before flight_date)
        start_date = flight_date - timedelta(days=15)
        formatted_start_date = start_date.strftime("%Y-%m-%d")
        
        # Make sure to include the flight date itself in the query
        # Add 1 day to make the query inclusive of the flight date
        end_date = flight_date + timedelta(days=1)
        formatted_flight_date = end_date.strftime("%Y-%m-%d")
        
        # Construct and execute the query to get data for the 15 days before flight date AND the flight date
        query = """
        SELECT FltNo, FltDate, Origin, Destination, ReportWeight, ReportVolume, OBW 
        FROM dbo.CapacityTransaction
        WHERE FltNo = ?
        AND Origin = ?
        AND Destination = ?
        AND FltDate >= ?
        AND FltDate < ?  -- Changed to < to work with the added day
        ORDER BY FltDate
        """
        
        cursor.execute(query, (flight_no, origin, destination, formatted_start_date, formatted_flight_date))
        
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
                
        # Calculate 15 days before flight date
        start_date = flight_date - timedelta(days=15)
        
        # Generate sample dates for the 15 days leading up to flight date INCLUDING the flight date
        sample_dates = [start_date + timedelta(days=i) for i in range(16)]  # 16 to include both start and end date
        
        # Make sure the last date is exactly the flight_date to ensure it's included
        if sample_dates[-1] != flight_date:
            sample_dates[-1] = flight_date
            
        # Make sure we have 16 dates (15 days before + flight date)
        num_dates = len(sample_dates)
        
        # Create sample data - ensure everything is a proper type
        sample_data = {
            'FltNo': [flight_no] * num_dates,
            'FltDate': sample_dates,
            'Origin': [origin] * num_dates,
            'Destination': [destination] * num_dates,
            'ReportWeight': [float(1000 + i * 50 + (100 * (i % 3))) for i in range(num_dates)],  # Ensure float
            'ReportVolume': [float(500 + i * 25 + (50 * (i % 4))) for i in range(num_dates)],    # Ensure float
            'OBW': [float(i * 2.5 - 10) for i in range(num_dates)]  # Ensure float
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
    
    # Get today's date and format it the same way as in the dataframe
    today = datetime.now().date()
    today_str = today.strftime('%d %b').upper()
    
    # Get departure date (from flight_date parameter) and format it
    if isinstance(flight_date, str):
        try:
            departure_date = datetime.strptime(flight_date.split('T')[0], "%Y-%m-%d").date()
        except:
            departure_date = datetime.now().date()
    else:
        departure_date = flight_date
    
    departure_str = departure_date.strftime('%d %b').upper()
    
    # Print for debugging
    print(f"Departure date: {departure_date}, formatted as: {departure_str}")
    print(f"Available dates in dataset: {df['FormattedDate'].unique()}")
    
    # Check if today is in the dataset
    today_in_dataset = today_str in df['FormattedDate'].values
    
    # Split data based on date (before today vs today and after)
    if today_in_dataset:
        # Convert FltDate to date objects for comparison
        df['DateOnly'] = df['FltDate'].dt.date
        
        # Create masks for past and future dates
        past_mask = df['DateOnly'] < today
        future_mask = df['DateOnly'] >= today
        
        # Split the dataframe
        past_df = df[past_mask]
        future_df = df[future_mask]
    else:
        # If today is not in the dataset, consider all data as future data
        past_df = pd.DataFrame(columns=df.columns)  # Empty DataFrame
        future_df = df
    
    # Calculate max OBW value and set y-axis max to 1000 higher
    max_obw = df['OBW'].max()
    y_axis_max = 1000 + max_obw
    
    # Create figure
    fig = go.Figure()
    
    # Add past weight bars (green)
    if not past_df.empty:
        fig.add_trace(
            go.Bar(
                x=past_df['FormattedDate'],
                y=past_df['ReportWeight'],
                name='Past Wt',
                marker_color='green',
                showlegend=True
            )
        )
        
        # Add past OBW line (red)
        fig.add_trace(
            go.Scatter(
                x=past_df['FormattedDate'],
                y=past_df['OBW'],
                name='Past OBW',
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=8, symbol='circle'),
                showlegend=True
            )
        )
    
    # Add future weight bars (blue)
    if not future_df.empty:
        fig.add_trace(
            go.Bar(
                x=future_df['FormattedDate'],
                y=future_df['ReportWeight'],
                name='Pred Wt',
                marker_color='blue',
                showlegend=True
            )
        )
        
        # Add future OBW line (orange)
        fig.add_trace(
            go.Scatter(
                x=future_df['FormattedDate'],
                y=future_df['OBW'],
                name='Pred OBW',
                mode='lines+markers',
                line=dict(color='orange', width=2),
                marker=dict(size=8, symbol='circle'),
                showlegend=True
            )
        )
    
    # Calculate the max y-value for vertical lines
    max_y = max(df['ReportWeight'].max(), df['OBW'].max()) * 1.1
    
    # Calculate the y-axis maximum (500 more than the height of "Today" annotation)
    y_axis_max = max_y + 500
    
    # Try to add today's line if today is in the dataset
    if today_in_dataset:
        try:
            fig.add_shape(
                type='line',
                x0=today_str,
                x1=today_str,
                y0=0,
                y1=max_y,
                line=dict(color='black', width=1.5, dash='dot'),
                xref='x',
                yref='y'
            )
            
            fig.add_annotation(
                x=today_str,
                y=max_y,
                text="Today",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40
            )
            
            # Add a dotted orange line connecting past and future OBW sections at today's date
            # This will be at the transition point between red and orange OBW lines
            if not past_df.empty and not future_df.empty:
                # Find the OBW values closest to today in both past and future dataframes
                past_today = past_df.loc[past_df['FltDate'] == past_df['FltDate'].max()]
                future_today = future_df.loc[future_df['FltDate'] == future_df['FltDate'].min()]
                
                if not past_today.empty and not future_today.empty:
                    # Get OBW values and dates for the connection
                    past_obw = past_today['OBW'].iloc[0]
                    future_obw = future_today['OBW'].iloc[0]
                    past_date = past_today['FormattedDate'].iloc[0]
                    future_date = future_today['FormattedDate'].iloc[0]
                    
                    # Add horizontal dotted orange line connecting the last point of past OBW to the first point of future OBW
                    fig.add_trace(
                        go.Scatter(
                            x=[past_date, future_date],
                            y=[past_obw, future_obw],
                            mode='lines',
                            line=dict(color='orange', width=2, dash='dot'),
                            showlegend=False
                        )
                    )
        except Exception as e:
            print(f"Error adding today's line or separator: {e}")
            # Skip adding today's line if there's an error
            pass
    else:
        # If today isn't in the dataset but we have both past and future data,
        # add the dotted orange line at the transition point
        if not past_df.empty and not future_df.empty:
            try:
                # Find the last date in past_df and first date in future_df
                transition_date_past = past_df.loc[past_df['FltDate'] == past_df['FltDate'].max()]
                transition_date_future = future_df.loc[future_df['FltDate'] == future_df['FltDate'].min()]
                
                if not transition_date_past.empty and not transition_date_future.empty:
                    # Get dates and OBW values for transition
                    transition_date_str_past = transition_date_past['FormattedDate'].iloc[0]
                    transition_date_str_future = transition_date_future['FormattedDate'].iloc[0]
                    past_obw = transition_date_past['OBW'].iloc[0]
                    future_obw = transition_date_future['OBW'].iloc[0]
                    
                    # Add dotted orange line between the two dates
                    fig.add_trace(
                        go.Scatter(
                            x=[transition_date_str_past, transition_date_str_future],
                            y=[past_obw, future_obw],
                            mode='lines',
                            line=dict(color='orange', width=2, dash='dot'),
                            showlegend=False
                        )
                    )
            except Exception as e:
                print(f"Error adding separator line: {e}")
                pass
    
    # Make sure departure date appears on x-axis (but don't add a vertical line)
    try:
        # If departure date is not in dataset, add it as a category on x-axis
        if departure_str not in df['FormattedDate'].values:
            print(f"Departure date {departure_str} not found in dataset, adding to x-axis")
            
            # Add an empty entry for this date to make it appear on the x-axis
            new_row = pd.DataFrame({
                'FltNo': [flight_no],
                'FltDate': [departure_date],
                'FormattedDate': [departure_str],
                'Origin': [origin],
                'Destination': [destination],
                'ReportWeight': [0],  # Zero values won't show a bar
                'ReportVolume': [0],
                'OBW': [0],
                'DateOnly': [departure_date]
            })
            
            # Add this to the appropriate dataframe (past or future)
            if departure_date < today:
                if not past_df.empty:
                    past_df = pd.concat([past_df, new_row], ignore_index=True)
                else:
                    past_df = new_row
            else:
                if not future_df.empty:
                    future_df = pd.concat([future_df, new_row], ignore_index=True)
                else:
                    future_df = new_row
            
            # Update the main dataframe
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Re-sort the dataframes by date
            if not past_df.empty:
                past_df = past_df.sort_values('FltDate')
            if not future_df.empty:
                future_df = future_df.sort_values('FltDate')
            df = df.sort_values('FltDate')
            
            # Update the figure with the new data
            # Clear existing traces
            fig.data = []
            
            # Recreate traces with updated data
            # Add past weight bars (green)
            if not past_df.empty:
                fig.add_trace(
                    go.Bar(
                        x=past_df['FormattedDate'],
                        y=past_df['ReportWeight'],
                        name='Past Weight (kg)',
                        marker_color='green',
                        showlegend=True
                    )
                )
                
                # Add past OBW line (red)
                fig.add_trace(
                    go.Scatter(
                        x=past_df['FormattedDate'],
                        y=past_df['OBW'],
                        name='Past OBW',
                        mode='lines+markers',
                        line=dict(color='red', width=2),
                        marker=dict(size=8, symbol='circle'),
                        showlegend=True
                    )
                )
            
            # Add future weight bars (blue)
            if not future_df.empty:
                fig.add_trace(
                    go.Bar(
                        x=future_df['FormattedDate'],
                        y=future_df['ReportWeight'],
                        name='Pred Weight (kg)',
                        marker_color='blue',
                        showlegend=True
                    )
                )
                
                # Add future OBW line (orange)
                fig.add_trace(
                    go.Scatter(
                        x=future_df['FormattedDate'],
                        y=future_df['OBW'],
                        name='Pred OBW',
                        mode='lines+markers',
                        line=dict(color='orange', width=2),
                        marker=dict(size=8, symbol='circle'),
                        showlegend=True
                    )
                )
    except Exception as e:
        print(f"Error processing departure date: {e}")
        # Skip if there's an error
        pass
    
    # Get all dates to ensure xaxis includes everything we need
    all_dates = sorted(list(set(df['FormattedDate'].values)))
    
    # Make sure departure_str is in the list
    if departure_str not in all_dates:
        all_dates.append(departure_str)
        all_dates.sort()  # Re-sort to keep in order
    
    # Update layout with single Y-axis and title in plot
    fig.update_layout(
        title=dict(
            text="Weight and Overbooking Status",
            x=0.5,  # Center title
            y=0.98  # Position near top
        ),
        xaxis=dict(
            title="Flight Date",
            tickmode='array',
            tickvals=all_dates,  # Show all dates including departure
            tickfont=dict(size=10),  # Smaller font for dates
            tickangle=45  # Angle the date labels to avoid overlap
        ),
        yaxis=dict(
            title="Weight (kg) / OBW",
            rangemode="tozero",  # Make y-axis start from zero
            range=[0, y_axis_max]  # Set range to 500 higher than the Today annotation height
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
        margin=dict(l=50, r=100, t=50, b=70),  # Increased bottom margin for angled date labels
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