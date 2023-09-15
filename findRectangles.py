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
    x_values = [point[0] for point in rectangle]
    y_values = [point[1] for point in rectangle]
    plt.plot(x_values + [x_values[0]], y_values + [y_values[0]], color='g', linewidth=2)

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
uploaded_file = st.file_uploader("Import File Containing Closed Curve", type=["txt"])
plot_all_points = st.checkbox("Plot All Points", value=False)
plot_lines_from_points = st.checkbox("Draw Lines", value=False)

if uploaded_file is not None:
    # Read the uploaded text file and split it into lines
    file_contents = uploaded_file.read().decode('utf-8').splitlines()

    # Initialize empty lists to store x and y coordinates
    x_values = []
    y_values = []

    # Parse each line to extract x and y coordinates separated by commas
    for line in file_contents:
        parts = line.strip().split(',')  # Assuming comma-separated values
        if len(parts) == 2:
            x_values.append(float(parts[0]))
            y_values.append(float(parts[1]))

    # Ensure the curve is closed by appending the first point to the end
    if len(x_values) >= 2:
        x_values.append(x_values[0])
        y_values.append(y_values[0])

    # Check if there are valid x and y values
    if x_values and y_values:
        # Apply spline interpolation to smooth the curve
        t = np.arange(len(x_values))
        fx = interp1d(t, x_values, kind='linear')
        fy = interp1d(t, y_values, kind='linear')

        t_smooth = np.linspace(0, len(x_values) - 1, 1000)
        x_smooth = fx(t_smooth)
        y_smooth = fy(t_smooth)

        # Plot the closed curve using Matplotlib
        plt.figure(figsize=(8, 6))
        if plot_all_points:
            # Plot all points if the checkbox is selected
            plt.scatter(x_values, y_values, marker='.', color='r')
        

        if plot_lines_from_points:
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
        st.error("The uploaded file does not contain valid data.")
else:
    st.info("Please upload a .txt file.")
