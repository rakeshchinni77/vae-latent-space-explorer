# Use a lightweight Python 3.10 base image for production builds.
FROM python:3.10-slim

# Set the application working directory inside the container.
WORKDIR /app

# Copy dependency definitions first to leverage Docker layer caching.
COPY requirements.txt ./

# Install Python dependencies without caching to keep the image smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project into the container.
COPY . .

# Expose the Streamlit port used by the application.
EXPOSE 8501

# Start the Streamlit application with the required host and port settings.
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
