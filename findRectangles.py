import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import random
from itertools import combinations

st.set_option('deprecation.showPyplotGlobalUse', False)

def find_bounding_box(coordinates):
    min_x = min(coordinates, key=lambda p: p[0])[0]
    max_x = max(coordinates, key=lambda p: p[0])[0]
    min_y = min(coordinates, key=lambda p: p[1])[1]
    max_y = max(coordinates, key=lambda p: p[1])[1]
    return min_x, max_x, min_y, max_y

def generate_rectangle_coordinates(coordinates, width_ratio=1.0, height_ratio=1.0):
    min_x, max_x, min_y, max_y = find_bounding_box(coordinates)

    width = (max_x - min_x) * width_ratio
    height = (max_y - min_y) * height_ratio

    # Calculate the coordinates of the four corners of the rectangle
    point1 = (min_x, min_y)
    point2 = (min_x + width, min_y)
    point3 = (min_x + width, min_y + height)
    point4 = (min_x, min_y + height)

    return point1, point2, point3, point4

def is_closed_curve(coordinates, tolerance=1e-6):
    if len(coordinates) < 3:
        return False

    start_point = coordinates[0]
    end_point = coordinates[-1]

    # Check if the start and end points are close enough to consider it closed
    return np.allclose(start_point, end_point, atol=tolerance)

def plot_rectangle(rectangle):
    x_values = [point[0] for point in rectangle + [rectangle[0]]]
    y_values = [point[1] for point in rectangle + [rectangle[0]]]
    plt.plot(x_values, y_values, color='g', linewidth=2)

def generate_valid_rectangles(coordinates):
    rectangles = []

    for p1, p2 in combinations(coordinates, 2):
        if p1[0] != p2[0] and p1[1] != p2[1]:
            p3 = (p2[0], p1[1])
            p4 = (p1[0], p2[1])
            if p3 in coordinates and p4 in coordinates:
                rectangles.append((p1, p2, p3, p4))

    return rectangles

st.title("Rectangle Finder")

uploaded_file = st.file_uploader("Import File Containing Closed Curve", type=["csv"])

plot_all_points = st.checkbox("Plot All Points", value=False)

if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Check if the DataFrame has columns named 'x' and 'y'
    if 'x' in df.columns and 'y' in df.columns:
        # Extract the unique x values and corresponding y values
        x = df['x'].values
        y = df['y'].values

        # Apply spline interpolation to smooth the curve
        t = np.arange(len(x))
        fx = interp1d(t, x, kind='cubic')
        fy = interp1d(t, y, kind='cubic')

        t_smooth = np.linspace(0, len(x) - 1, 1000)
        x_smooth = fx(t_smooth)
        y_smooth = fy(t_smooth)

        # Plot the closed curve using Matplotlib
        plt.figure(figsize=(8, 6))
        if plot_all_points:
            # Plot all points if the checkbox is selected
            plt.scatter(df['x'], df['y'], marker='.', color='r')
        plt.plot(x_smooth, y_smooth, color='b')

        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.title('Closed Curve')
        plt.gca().set_aspect('equal', adjustable='box')

        # Display the plot in Streamlit
        st.pyplot()

        # Check if the curve is closed
        if is_closed_curve(list(zip(x_smooth, y_smooth))):
            st.write("The curve is closed.")

            coordinates = list(zip(x_smooth, y_smooth))

            # Generate valid rectangles
            valid_rectangles = generate_valid_rectangles(coordinates)

            # Choose one random valid rectangle if any
            if valid_rectangles:
                chosen_rectangle = random.choice(valid_rectangles)
                for point in chosen_rectangle:
                    st.write(point)
                plot_rectangle(chosen_rectangle)  # Plot the chosen rectangle
            else:
                st.error("No valid rectangles can be formed.")
        else:
            st.error("The uploaded curve is not closed.")
    else:
        st.error("The uploaded file does not contain 'x' and 'y' columns.")
else:
    st.info("Please upload a CSV file.")
