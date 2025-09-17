# Use slim Python base
FROM python:3.11-slim

# Set working dir
WORKDIR /app

# Install system packages (for audio)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (caching layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the Streamlit port
EXPOSE 8080

# Streamlit tweaks for AWS App Runner
ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLEXSRSFPROTECTION=false
ENV STREAMLIT_SERVER_BASEURLPATH=/
ENV STREAMLIT_SERVER_ENABLEWEBSOCKETCOMPRESSION=false
ENV STREAMLIT_SERVER_ENABLEWEBSOCKETPING=false
ENV STREAMLIT_BROWSER_SERVERADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_SERVERPORT=8080

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]

